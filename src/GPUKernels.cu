#include "GPUKernels.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <stdio.h>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            return false; \
        } \
    } while(0)

// Device function: Haversine distance calculation
__device__ double haversineDistanceGPU(double lat1, double lon1, double lat2, double lon2) {
    const double R = 6371.0; // Earth radius in km
    const double DEG_TO_RAD = M_PI / 180.0;
    
    double dLat = (lat2 - lat1) * DEG_TO_RAD;
    double dLon = (lon2 - lon1) * DEG_TO_RAD;
    
    double a = sin(dLat/2.0) * sin(dLat/2.0) +
               cos(lat1 * DEG_TO_RAD) * cos(lat2 * DEG_TO_RAD) *
               sin(dLon/2.0) * sin(dLon/2.0);
    
    double c = 2.0 * atan2(sqrt(a), sqrt(1.0-a));
    return R * c;
}

// Device function: Convert km to lat/lon
__device__ void kmToLatLonGPU(double x, double y, double refLat, double refLon, 
                              double &lat, double &lon) {
    const double DEG_TO_RAD = M_PI / 180.0;
    double latRad = refLat * DEG_TO_RAD;
    lon = refLon + x / (111.320 * cos(latRad));
    lat = refLat + y / 110.574;
}

// Device function: Get velocity (homogeneous model)
__device__ double getVelocityGPU(double z, double homogeneousVp, 
                                 const VelocityLayer1D_GPU *layers1D, int nLayers) {
    if (nLayers == 0) {
        return homogeneousVp;
    }
    
    // 1D model
    for (int i = 0; i < nLayers; i++) {
        if (z <= layers1D[i].maxDepth) {
            return layers1D[i].vp;
        }
    }
    
    // Beyond all layers, use last layer velocity
    return layers1D[nLayers-1].vp;
}

// Device function: Calculate travel time
__device__ double calculateTravelTimeGPU(double x, double y, double z,
                                        double stationLat, double stationLon,
                                        double refLat, double refLon,
                                        double homogeneousVp,
                                        const VelocityLayer1D_GPU *layers1D, 
                                        int nLayers) {
    // Convert event location to lat/lon
    double eventLat, eventLon;
    kmToLatLonGPU(x, y, refLat, refLon, eventLat, eventLon);
    
    // Calculate horizontal distance
    double horizDist = haversineDistanceGPU(eventLat, eventLon, stationLat, stationLon);
    
    // Calculate 3D distance
    double dist3D = sqrt(horizDist * horizDist + z * z);
    
    // Get average velocity
    double avgVelocity = getVelocityGPU(z/2.0, homogeneousVp, layers1D, nLayers);
    
    return dist3D / avgVelocity;
}

// Device function: Calculate misfit for single location
__device__ double calculateMisfitGPU(double x, double y, double z, double t0,
                                    const StationData_GPU *stations, int nStations,
                                    const double *observedTimes,
                                    double refLat, double refLon,
                                    double homogeneousVp,
                                    const VelocityLayer1D_GPU *layers1D,
                                    int nLayers) {
    double sumSquaredResiduals = 0.0;
    
    for (int i = 0; i < nStations; i++) {
        double travelTime = calculateTravelTimeGPU(x, y, z,
                                                   stations[i].latitude,
                                                   stations[i].longitude,
                                                   refLat, refLon,
                                                   homogeneousVp,
                                                   layers1D, nLayers);
        double predicted = t0 + travelTime;
        double residual = observedTimes[i] - predicted;
        sumSquaredResiduals += residual * residual;
    }
    
    return sqrt(sumSquaredResiduals / nStations);
}

// CUDA Kernel: Grid Search
__global__ void gridSearchKernel(const GridSearchParams params,
                                const StationData_GPU *stations,
                                const double *observedTimes,
                                const VelocityLayer1D_GPU *layers1D,
                                double *misfits,
                                int *bestIndices) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPoints = params.nX * params.nY * params.nZ;
    
    if (idx >= totalPoints) return;
    
    // Calculate 3D grid position
    int iz = idx / (params.nX * params.nY);
    int temp = idx % (params.nX * params.nY);
    int iy = temp / params.nX;
    int ix = temp % params.nX;
    
    // Calculate actual coordinates
    double x = params.xMin + ix * params.gridSpacing;
    double y = params.yMin + iy * params.gridSpacing;
    double z = params.depthMin + iz * params.gridSpacing;
    
    // Calculate misfit for this location
    double t0 = 0.0; // Initial estimate
    double misfit = calculateMisfitGPU(x, y, z, t0,
                                      stations, params.nStations,
                                      observedTimes,
                                      params.refLat, params.refLon,
                                      params.homogeneousVp,
                                      layers1D, params.nLayers);
    
    // Store result
    misfits[idx] = misfit;
    
    // Atomic minimum to find best fit (use shared memory for block-level reduction)
    __shared__ double sharedMisfit[256];
    __shared__ int sharedIdx[256];
    
    int tid = threadIdx.x;
    sharedMisfit[tid] = misfit;
    sharedIdx[tid] = idx;
    __syncthreads();
    
    // Block-level reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sharedMisfit[tid + stride] < sharedMisfit[tid]) {
                sharedMisfit[tid] = sharedMisfit[tid + stride];
                sharedIdx[tid] = sharedIdx[tid + stride];
            }
        }
        __syncthreads();
    }
    
    // Thread 0 writes block result
    if (tid == 0) {
        atomicMin((int*)&misfits[0], __float_as_int((float)sharedMisfit[0]));
        if (misfits[0] == sharedMisfit[0]) {
            bestIndices[0] = sharedIdx[0];
        }
    }
}

// CUDA Kernel: Monte Carlo Grid Search
__global__ void monteCarloSearchKernel(const GridSearchParams params,
                                      const StationData_GPU *stations,
                                      const double *observedTimes,
                                      const VelocityLayer1D_GPU *layers1D,
                                      unsigned int seed,
                                      int nSamples,
                                      double *misfits,
                                      double *bestResults) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= nSamples) return;
    
    // Simple LCG random number generator
    unsigned int state = seed + idx;
    
    auto lcg = [&state]() -> double {
        state = (1103515245u * state + 12345u);
        return (state & 0x7FFFFFFFu) / 2147483647.0;
    };
    
    // Random location within bounds
    double x = params.xMin + lcg() * (params.xMax - params.xMin);
    double y = params.yMin + lcg() * (params.yMax - params.yMin);
    double z = params.depthMin + lcg() * (params.depthMax - params.depthMin);
    
    // Calculate misfit
    double t0 = 0.0;
    double misfit = calculateMisfitGPU(x, y, z, t0,
                                      stations, params.nStations,
                                      observedTimes,
                                      params.refLat, params.refLon,
                                      params.homogeneousVp,
                                      layers1D, params.nLayers);
    
    misfits[idx] = misfit;
    
    // Store coordinates
    bestResults[idx * 4 + 0] = x;
    bestResults[idx * 4 + 1] = y;
    bestResults[idx * 4 + 2] = z;
    bestResults[idx * 4 + 3] = misfit;
}

// CUDA Kernel: Calculate Jacobian (finite differences)
__global__ void calculateJacobianKernel(const GridSearchParams params,
                                       const StationData_GPU *stations,
                                       const double *observedTimes,
                                       const VelocityLayer1D_GPU *layers1D,
                                       double x, double y, double z, double t0,
                                       double h,
                                       double *jacobian) {
    int stationIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (stationIdx >= params.nStations) return;
    
    // Calculate partial derivatives using finite differences
    // J[i][0] = dT/dx
    double tt_x1 = calculateTravelTimeGPU(x + h, y, z,
                                         stations[stationIdx].latitude,
                                         stations[stationIdx].longitude,
                                         params.refLat, params.refLon,
                                         params.homogeneousVp,
                                         layers1D, params.nLayers);
    
    double tt_x0 = calculateTravelTimeGPU(x - h, y, z,
                                         stations[stationIdx].latitude,
                                         stations[stationIdx].longitude,
                                         params.refLat, params.refLon,
                                         params.homogeneousVp,
                                         layers1D, params.nLayers);
    
    jacobian[stationIdx * 4 + 0] = -(tt_x1 - tt_x0) / (2.0 * h);
    
    // J[i][1] = dT/dy
    double tt_y1 = calculateTravelTimeGPU(x, y + h, z,
                                         stations[stationIdx].latitude,
                                         stations[stationIdx].longitude,
                                         params.refLat, params.refLon,
                                         params.homogeneousVp,
                                         layers1D, params.nLayers);
    
    double tt_y0 = calculateTravelTimeGPU(x, y - h, z,
                                         stations[stationIdx].latitude,
                                         stations[stationIdx].longitude,
                                         params.refLat, params.refLon,
                                         params.homogeneousVp,
                                         layers1D, params.nLayers);
    
    jacobian[stationIdx * 4 + 1] = -(tt_y1 - tt_y0) / (2.0 * h);
    
    // J[i][2] = dT/dz
    double tt_z1 = calculateTravelTimeGPU(x, y, z + h,
                                         stations[stationIdx].latitude,
                                         stations[stationIdx].longitude,
                                         params.refLat, params.refLon,
                                         params.homogeneousVp,
                                         layers1D, params.nLayers);
    
    double tt_z0 = calculateTravelTimeGPU(x, y, z - h,
                                         stations[stationIdx].latitude,
                                         stations[stationIdx].longitude,
                                         params.refLat, params.refLon,
                                         params.homogeneousVp,
                                         layers1D, params.nLayers);
    
    jacobian[stationIdx * 4 + 2] = -(tt_z1 - tt_z0) / (2.0 * h);
    
    // J[i][3] = dT/dt0 = -1
    jacobian[stationIdx * 4 + 3] = -1.0;
}

// Host function: Initialize GPU
bool initializeGPU(GPUDeviceInfo &info) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        info.available = false;
        info.deviceName = "No CUDA-capable device found";
        return false;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    info.available = true;
    info.deviceName = std::string(prop.name);
    info.computeCapability = prop.major * 10 + prop.minor;
    info.totalMemory = prop.totalGlobalMem;
    info.maxThreadsPerBlock = prop.maxThreadsPerBlock;
    info.multiProcessorCount = prop.multiProcessorCount;
    
    return true;
}

// Host function: Launch Grid Search on GPU
bool launchGridSearchGPU(const GridSearchParams &params,
                        const StationData_GPU *h_stations,
                        const double *h_observedTimes,
                        const VelocityLayer1D_GPU *h_layers1D,
                        GridSearchResult &result,
                        ProgressCallback callback) {
    // Allocate device memory
    StationData_GPU *d_stations;
    double *d_observedTimes;
    VelocityLayer1D_GPU *d_layers1D;
    double *d_misfits;
    int *d_bestIndices;
    
    int totalPoints = params.nX * params.nY * params.nZ;
    
    CUDA_CHECK(cudaMalloc(&d_stations, params.nStations * sizeof(StationData_GPU)));
    CUDA_CHECK(cudaMalloc(&d_observedTimes, params.nStations * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_layers1D, params.nLayers * sizeof(VelocityLayer1D_GPU)));
    CUDA_CHECK(cudaMalloc(&d_misfits, totalPoints * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_bestIndices, sizeof(int)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_stations, h_stations, 
                         params.nStations * sizeof(StationData_GPU), 
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_observedTimes, h_observedTimes, 
                         params.nStations * sizeof(double), 
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_layers1D, h_layers1D, 
                         params.nLayers * sizeof(VelocityLayer1D_GPU), 
                         cudaMemcpyHostToDevice));
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalPoints + threadsPerBlock - 1) / threadsPerBlock;
    
    if (callback) {
        callback(0, "Launching GPU Grid Search...");
    }
    
    gridSearchKernel<<<blocksPerGrid, threadsPerBlock>>>(
        params, d_stations, d_observedTimes, d_layers1D, d_misfits, d_bestIndices
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    if (callback) {
        callback(50, "GPU computation complete, retrieving results...");
    }
    
    // Copy results back
    double *h_misfits = new double[totalPoints];
    int bestIdx;
    
    CUDA_CHECK(cudaMemcpy(h_misfits, d_misfits, totalPoints * sizeof(double), 
                         cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&bestIdx, d_bestIndices, sizeof(int), 
                         cudaMemcpyDeviceToHost));
    
    // Find best result
    double bestMisfit = h_misfits[0];
    bestIdx = 0;
    for (int i = 1; i < totalPoints; i++) {
        if (h_misfits[i] < bestMisfit) {
            bestMisfit = h_misfits[i];
            bestIdx = i;
        }
    }
    
    // Calculate coordinates
    int iz = bestIdx / (params.nX * params.nY);
    int temp = bestIdx % (params.nX * params.nY);
    int iy = temp / params.nX;
    int ix = temp % params.nX;
    
    result.x = params.xMin + ix * params.gridSpacing;
    result.y = params.yMin + iy * params.gridSpacing;
    result.z = params.depthMin + iz * params.gridSpacing;
    result.misfit = bestMisfit;
    result.converged = true;
    
    // Cleanup
    delete[] h_misfits;
    cudaFree(d_stations);
    cudaFree(d_observedTimes);
    cudaFree(d_layers1D);
    cudaFree(d_misfits);
    cudaFree(d_bestIndices);
    
    if (callback) {
        callback(100, "GPU Grid Search complete!");
    }
    
    return true;
}

// Host function: Launch Monte Carlo Search on GPU
bool launchMonteCarloSearchGPU(const GridSearchParams &params,
                              const StationData_GPU *h_stations,
                              const double *h_observedTimes,
                              const VelocityLayer1D_GPU *h_layers1D,
                              int nSamples,
                              GridSearchResult &result,
                              ProgressCallback callback) {
    // Allocate device memory
    StationData_GPU *d_stations;
    double *d_observedTimes;
    VelocityLayer1D_GPU *d_layers1D;
    double *d_misfits;
    double *d_bestResults;
    
    CUDA_CHECK(cudaMalloc(&d_stations, params.nStations * sizeof(StationData_GPU)));
    CUDA_CHECK(cudaMalloc(&d_observedTimes, params.nStations * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_layers1D, params.nLayers * sizeof(VelocityLayer1D_GPU)));
    CUDA_CHECK(cudaMalloc(&d_misfits, nSamples * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_bestResults, nSamples * 4 * sizeof(double)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_stations, h_stations, 
                         params.nStations * sizeof(StationData_GPU), 
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_observedTimes, h_observedTimes, 
                         params.nStations * sizeof(double), 
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_layers1D, h_layers1D, 
                         params.nLayers * sizeof(VelocityLayer1D_GPU), 
                         cudaMemcpyHostToDevice));
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (nSamples + threadsPerBlock - 1) / threadsPerBlock;
    
    unsigned int seed = static_cast<unsigned int>(time(nullptr));
    
    if (callback) {
        callback(0, "Launching GPU Monte Carlo Search...");
    }
    
    monteCarloSearchKernel<<<blocksPerGrid, threadsPerBlock>>>(
        params, d_stations, d_observedTimes, d_layers1D, 
        seed, nSamples, d_misfits, d_bestResults
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back
    double *h_bestResults = new double[nSamples * 4];
    CUDA_CHECK(cudaMemcpy(h_bestResults, d_bestResults, 
                         nSamples * 4 * sizeof(double), 
                         cudaMemcpyDeviceToHost));
    
    // Find best result
    double bestMisfit = h_bestResults[3];
    int bestIdx = 0;
    for (int i = 1; i < nSamples; i++) {
        if (h_bestResults[i * 4 + 3] < bestMisfit) {
            bestMisfit = h_bestResults[i * 4 + 3];
            bestIdx = i;
        }
    }
    
    result.x = h_bestResults[bestIdx * 4 + 0];
    result.y = h_bestResults[bestIdx * 4 + 1];
    result.z = h_bestResults[bestIdx * 4 + 2];
    result.misfit = bestMisfit;
    result.converged = true;
    
    // Cleanup
    delete[] h_bestResults;
    cudaFree(d_stations);
    cudaFree(d_observedTimes);
    cudaFree(d_layers1D);
    cudaFree(d_misfits);
    cudaFree(d_bestResults);
    
    if (callback) {
        callback(100, "GPU Monte Carlo Search complete!");
    }
    
    return true;
}
