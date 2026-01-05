#include "CUDACommon.cuh"
#include <cfloat>

// Kernel for exhaustive grid search
__global__ void gridSearchKernel(
    const StationGPU* stations,
    int numStations,
    float xMin, float xMax, int nX,
    float yMin, float yMax, int nY,
    float zMin, float zMax, int nZ,
    float gridSpacing,
    float homogeneousVp,
    float* misfitGrid,
    float* bestParams,  // [x, y, z, t0, misfit]
    float* minArrival)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPoints = nX * nY * nZ;
    
    if (idx >= totalPoints) return;
    
    // Convert linear index to 3D grid coordinates
    int iz = idx / (nX * nY);
    int remainder = idx % (nX * nY);
    int iy = remainder / nX;
    int ix = remainder % nX;
    
    // Compute grid point coordinates
    float x = xMin + ix * gridSpacing;
    float y = yMin + iy * gridSpacing;
    float z = zMin + iz * gridSpacing;
    
    // Estimate origin time from minimum arrival
    float t0 = *minArrival - 50.0f;
    
    // Compute misfit
    float misfit = computeMisfitDevice(x, y, z, t0, stations, numStations, homogeneousVp);
    
    // Store misfit in grid
    misfitGrid[idx] = misfit;
    
    // Atomic update for best solution
    // Use atomicMin with integer comparison for thread-safety
    unsigned int* misfit_as_uint = (unsigned int*)&bestParams[4];
    unsigned int old_uint = *misfit_as_uint;
    unsigned int assumed_uint;
    
    do {
        assumed_uint = old_uint;
        float old_misfit = __uint_as_float(assumed_uint);
        
        if (misfit < old_misfit) {
            unsigned int new_uint = __float_as_uint(misfit);
            old_uint = atomicCAS(misfit_as_uint, assumed_uint, new_uint);
            
            // If successful, update parameters
            if (old_uint == assumed_uint) {
                bestParams[0] = x;
                bestParams[1] = y;
                bestParams[2] = z;
                bestParams[3] = t0;
            }
        } else {
            break;
        }
    } while (old_uint != assumed_uint);
}

// Kernel for Monte Carlo random sampling
__global__ void monteCarloSearchKernel(
    const StationGPU* stations,
    int numStations,
    float xMin, float xMax,
    float yMin, float yMax,
    float zMin, float zMax,
    float homogeneousVp,
    float* bestParams,  // [x, y, z, t0, misfit]
    float* minArrival,
    unsigned int* randomSeeds,
    int numSamples)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= numSamples) return;
    
    // Simple LCG random number generator
    unsigned int seed = randomSeeds[idx];
    
    auto lcg_rand = [&seed]() -> float {
        seed = (1103515245u * seed + 12345u) & 0x7fffffffu;
        return (float)seed / 0x7fffffffu;
    };
    
    // Generate random point
    float x = xMin + lcg_rand() * (xMax - xMin);
    float y = yMin + lcg_rand() * (yMax - yMin);
    float z = zMin + lcg_rand() * (zMax - zMin);
    float t0 = *minArrival + lcg_rand() * 100.0f - 100.0f;
    
    // Compute misfit
    float misfit = computeMisfitDevice(x, y, z, t0, stations, numStations, homogeneousVp);
    
    // Atomic update for best solution
    unsigned int* misfit_as_uint = (unsigned int*)&bestParams[4];
    unsigned int old_uint = *misfit_as_uint;
    unsigned int assumed_uint;
    
    do {
        assumed_uint = old_uint;
        float old_misfit = __uint_as_float(assumed_uint);
        
        if (misfit < old_misfit) {
            unsigned int new_uint = __float_as_uint(misfit);
            old_uint = atomicCAS(misfit_as_uint, assumed_uint, new_uint);
            
            if (old_uint == assumed_uint) {
                bestParams[0] = x;
                bestParams[1] = y;
                bestParams[2] = z;
                bestParams[3] = t0;
            }
        } else {
            break;
        }
    } while (old_uint != assumed_uint);
}

// =============================================================================
// Host Functions (C++ interface)
// =============================================================================

extern "C" {

void cudaGridSearch(
    const float* h_stations,  // [numStations * 3] (lat, lon, arrivalTime)
    int numStations,
    float xMin, float xMax, int nX,
    float yMin, float yMax, int nY,
    float zMin, float zMax, int nZ,
    float gridSpacing,
    float homogeneousVp,
    float* h_bestParams,  // Output: [x, y, z, t0, misfit]
    float* h_misfitGrid)  // Optional: full misfit grid
{
    // Allocate device memory
    StationGPU* d_stations;
    float* d_misfitGrid;
    float* d_bestParams;
    float* d_minArrival;
    
    int totalPoints = nX * nY * nZ;
    size_t stationsSize = numStations * sizeof(StationGPU);
    size_t gridSize = totalPoints * sizeof(float);
    
    cudaMalloc(&d_stations, stationsSize);
    cudaMalloc(&d_misfitGrid, gridSize);
    cudaMalloc(&d_bestParams, 5 * sizeof(float));
    cudaMalloc(&d_minArrival, sizeof(float));
    
    // Copy stations to device
    StationGPU* h_stationsGPU = new StationGPU[numStations];
    for (int i = 0; i < numStations; ++i) {
        h_stationsGPU[i].lat = h_stations[i * 3 + 0];
        h_stationsGPU[i].lon = h_stations[i * 3 + 1];
        h_stationsGPU[i].arrivalTime = h_stations[i * 3 + 2];
    }
    cudaMemcpy(d_stations, h_stationsGPU, stationsSize, cudaMemcpyHostToDevice);
    
    // Find minimum arrival time
    float minArr = FLT_MAX;
    for (int i = 0; i < numStations; ++i) {
        if (h_stationsGPU[i].arrivalTime < minArr) {
            minArr = h_stationsGPU[i].arrivalTime;
        }
    }
    cudaMemcpy(d_minArrival, &minArr, sizeof(float), cudaMemcpyHostToDevice);
    
    // Initialize best params with worst values
    float initBest[5] = {0, 0, 0, 0, FLT_MAX};
    cudaMemcpy(d_bestParams, initBest, 5 * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalPoints + threadsPerBlock - 1) / threadsPerBlock;
    
    gridSearchKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_stations, numStations,
        xMin, xMax, nX,
        yMin, yMax, nY,
        zMin, zMax, nZ,
        gridSpacing,
        homogeneousVp,
        d_misfitGrid,
        d_bestParams,
        d_minArrival
    );
    
    // Wait for kernel to finish
    cudaDeviceSynchronize();
    
    // Copy results back
    cudaMemcpy(h_bestParams, d_bestParams, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    
    if (h_misfitGrid != nullptr) {
        cudaMemcpy(h_misfitGrid, d_misfitGrid, gridSize, cudaMemcpyDeviceToHost);
    }
    
    // Cleanup
    cudaFree(d_stations);
    cudaFree(d_misfitGrid);
    cudaFree(d_bestParams);
    cudaFree(d_minArrival);
    delete[] h_stationsGPU;
}

void cudaMonteCarloSearch(
    const float* h_stations,
    int numStations,
    float xMin, float xMax,
    float yMin, float yMax,
    float zMin, float zMax,
    float homogeneousVp,
    int numSamples,
    float* h_bestParams)
{
    // Allocate device memory
    StationGPU* d_stations;
    float* d_bestParams;
    float* d_minArrival;
    unsigned int* d_randomSeeds;
    
    size_t stationsSize = numStations * sizeof(StationGPU);
    
    cudaMalloc(&d_stations, stationsSize);
    cudaMalloc(&d_bestParams, 5 * sizeof(float));
    cudaMalloc(&d_minArrival, sizeof(float));
    cudaMalloc(&d_randomSeeds, numSamples * sizeof(unsigned int));
    
    // Copy stations
    StationGPU* h_stationsGPU = new StationGPU[numStations];
    for (int i = 0; i < numStations; ++i) {
        h_stationsGPU[i].lat = h_stations[i * 3 + 0];
        h_stationsGPU[i].lon = h_stations[i * 3 + 1];
        h_stationsGPU[i].arrivalTime = h_stations[i * 3 + 2];
    }
    cudaMemcpy(d_stations, h_stationsGPU, stationsSize, cudaMemcpyHostToDevice);
    
    // Find min arrival
    float minArr = FLT_MAX;
    for (int i = 0; i < numStations; ++i) {
        if (h_stationsGPU[i].arrivalTime < minArr) {
            minArr = h_stationsGPU[i].arrivalTime;
        }
    }
    cudaMemcpy(d_minArrival, &minArr, sizeof(float), cudaMemcpyHostToDevice);
    
    // Initialize random seeds
    unsigned int* h_seeds = new unsigned int[numSamples];
    for (int i = 0; i < numSamples; ++i) {
        h_seeds[i] = (unsigned int)(i * 1103515245u + 12345u);
    }
    cudaMemcpy(d_randomSeeds, h_seeds, numSamples * sizeof(unsigned int), cudaMemcpyHostToDevice);
    
    // Initialize best params
    float initBest[5] = {0, 0, 0, 0, FLT_MAX};
    cudaMemcpy(d_bestParams, initBest, 5 * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numSamples + threadsPerBlock - 1) / threadsPerBlock;
    
    monteCarloSearchKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_stations, numStations,
        xMin, xMax, yMin, yMax, zMin, zMax,
        homogeneousVp,
        d_bestParams,
        d_minArrival,
        d_randomSeeds,
        numSamples
    );
    
    // Wait and copy results
    cudaDeviceSynchronize();
    cudaMemcpy(h_bestParams, d_bestParams, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_stations);
    cudaFree(d_bestParams);
    cudaFree(d_minArrival);
    cudaFree(d_randomSeeds);
    delete[] h_stationsGPU;
    delete[] h_seeds;
}

} // extern "C"
