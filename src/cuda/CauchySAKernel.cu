#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cfloat>
#include <cmath>

__constant__ float DEG_TO_RAD = 3.14159265f / 180.0f;
__constant__ float EARTH_RADIUS = 6371.0f;
__constant__ float PI = 3.14159265f;

struct StationGPU {
    float lat;
    float lon;
    float arrivalTime;
};

struct CauchyParams {
    float T0;
    float Tf;
    float alpha;
    int dimension;
    float initialStepSize;
    int iterPerTemp;
    int maxIter;
    int coolingSchedule; // 0=Cauchy, 1=FastCauchy, 2=VeryFast
};

// =============================================================================
// Device Functions
// =============================================================================

__device__ float haversineDistance(float lat1, float lon1, float lat2, float lon2) {
    float dLat = (lat2 - lat1) * DEG_TO_RAD;
    float dLon = (lon2 - lon1) * DEG_TO_RAD;
    
    lat1 *= DEG_TO_RAD;
    lat2 *= DEG_TO_RAD;
    
    float a = sinf(dLat/2) * sinf(dLat/2) +
              cosf(lat1) * cosf(lat2) * sinf(dLon/2) * sinf(dLon/2);
    float c = 2 * atan2f(sqrtf(a), sqrtf(1-a));
    
    return EARTH_RADIUS * c;
}

__device__ float computeMisfitGPU(
    float x, float y, float z, float t0,
    const StationGPU* stations, int numStations,
    float homogeneousVp)
{
    float misfit = 0.0f;
    
    for (int i = 0; i < numStations; ++i) {
        float epicentralDist = haversineDistance(y, x, stations[i].lat, stations[i].lon);
        float distance3D = sqrtf(epicentralDist * epicentralDist + z * z);
        float travelTime = distance3D / homogeneousVp;
        float predictedArrival = t0 + travelTime;
        float residual = stations[i].arrivalTime - predictedArrival;
        misfit += residual * residual;
    }
    
    return sqrtf(misfit / numStations);
}

__device__ float updateTemperatureCauchy(float T, int iteration, const CauchyParams& params) {
    switch(params.coolingSchedule) {
        case 0: // Cauchy: T(k) = T0 / (1 + k)
            return params.T0 / (1.0f + iteration / params.iterPerTemp);
            
        case 1: // Fast Cauchy: T(k) = T0 / (1 + α×k)
            return params.T0 / (1.0f + params.alpha * (iteration / params.iterPerTemp));
            
        case 2: // Very Fast: T(k) = T0 × exp(-α × k^(1/D))
            return params.T0 * expf(-params.alpha * powf(iteration / params.iterPerTemp, 1.0f / params.dimension));
            
        default:
            return params.T0 / (1.0f + iteration / params.iterPerTemp);
    }
}

// Generate Cauchy-distributed random number
__device__ float cauchyRandom(curandState* state) {
    // Cauchy distribution: tan(π(u - 0.5))
    float u = curand_uniform(state);
    return tanf(PI * (u - 0.5f));
}

__device__ bool cauchyAcceptance(float currentMisfit, float newMisfit, float T, curandState* state) {
    if (newMisfit < currentMisfit) {
        return true;
    }
    
    // Generalized acceptance with Cauchy distribution
    float deltaE = newMisfit - currentMisfit;
    
    // Cauchy acceptance: 1 / (1 + (ΔE/T)²)
    float probability = 1.0f / (1.0f + (deltaE / T) * (deltaE / T));
    float random = curand_uniform(state);
    
    return random < probability;
}

// =============================================================================
// Kernels
// =============================================================================

__global__ void initRandomStates(curandState* states, unsigned long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void cauchySAKernel(
    const StationGPU* stations,
    int numStations,
    float xMin, float xMax,
    float yMin, float yMax,
    float zMin, float zMax,
    float homogeneousVp,
    CauchyParams params,
    curandState* states,
    float* solutions,
    float* misfitHistory,
    int numChains)
{
    int chainIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (chainIdx >= numChains) return;
    
    curandState localState = states[chainIdx];
    
    // Initialize random solution
    float currentX = xMin + curand_uniform(&localState) * (xMax - xMin);
    float currentY = yMin + curand_uniform(&localState) * (yMax - yMin);
    float currentZ = zMin + curand_uniform(&localState) * (zMax - zMin);
    float currentT0 = curand_uniform(&localState) * 100.0f - 50.0f;
    
    float currentMisfit = computeMisfitGPU(currentX, currentY, currentZ, currentT0,
                                           stations, numStations, homogeneousVp);
    
    float bestX = currentX;
    float bestY = currentY;
    float bestZ = currentZ;
    float bestT0 = currentT0;
    float bestMisfit = currentMisfit;
    
    float T = params.T0;
    float stepSize = params.initialStepSize;
    
    // Main Cauchy SA loop
    for (int iter = 0; iter < params.maxIter; ++iter) {
        // Cauchy annealing iterations at current temperature
        for (int tempIter = 0; tempIter < params.iterPerTemp; ++tempIter) {
            // Generate neighbor using Cauchy distribution
            // Cauchy distribution has heavier tails than Gaussian
            // This allows for larger jumps, helping escape local minima
            
            float cauchyX = cauchyRandom(&localState);
            float cauchyY = cauchyRandom(&localState);
            float cauchyZ = cauchyRandom(&localState);
            float cauchyT0 = cauchyRandom(&localState);
            
            // Scale by temperature and step size
            float tempFactor = T / params.T0;
            float scaleX = (xMax - xMin) * stepSize * tempFactor;
            float scaleY = (yMax - yMin) * stepSize * tempFactor;
            float scaleZ = (zMax - zMin) * stepSize * tempFactor;
            float scaleT0 = 10.0f * stepSize * tempFactor;
            
            // Limit Cauchy to avoid extremely large jumps
            cauchyX = fmaxf(-3.0f, fminf(3.0f, cauchyX));
            cauchyY = fmaxf(-3.0f, fminf(3.0f, cauchyY));
            cauchyZ = fmaxf(-3.0f, fminf(3.0f, cauchyZ));
            cauchyT0 = fmaxf(-3.0f, fminf(3.0f, cauchyT0));
            
            float newX = currentX + cauchyX * scaleX;
            float newY = currentY + cauchyY * scaleY;
            float newZ = currentZ + cauchyZ * scaleZ;
            float newT0 = currentT0 + cauchyT0 * scaleT0;
            
            // Apply boundary constraints
            newX = fmaxf(xMin, fminf(xMax, newX));
            newY = fmaxf(yMin, fminf(yMax, newY));
            newZ = fmaxf(zMin, fminf(zMax, newZ));
            
            // Compute new misfit
            float newMisfit = computeMisfitGPU(newX, newY, newZ, newT0,
                                              stations, numStations, homogeneousVp);
            
            // Cauchy acceptance criterion
            if (cauchyAcceptance(currentMisfit, newMisfit, T, &localState)) {
                currentX = newX;
                currentY = newY;
                currentZ = newZ;
                currentT0 = newT0;
                currentMisfit = newMisfit;
                
                // Update best
                if (newMisfit < bestMisfit) {
                    bestX = newX;
                    bestY = newY;
                    bestZ = newZ;
                    bestT0 = newT0;
                    bestMisfit = newMisfit;
                }
            }
        }
        
        // Update temperature (slower cooling for Cauchy)
        T = updateTemperatureCauchy(T, iter, params);
        
        // Adaptive step size based on temperature
        stepSize = params.initialStepSize * (T / params.T0);
        
        // Store misfit history
        if (iter < params.maxIter) {
            misfitHistory[chainIdx * params.maxIter + iter] = bestMisfit;
        }
        
        // Early stopping
        if (T < params.Tf) break;
    }
    
    // Store final solution
    solutions[chainIdx * 5 + 0] = bestX;
    solutions[chainIdx * 5 + 1] = bestY;
    solutions[chainIdx * 5 + 2] = bestZ;
    solutions[chainIdx * 5 + 3] = bestT0;
    solutions[chainIdx * 5 + 4] = bestMisfit;
    
    states[chainIdx] = localState;
}

// =============================================================================
// Host Interface
// =============================================================================

extern "C" {

void cudaCauchySA(
    const float* h_stations,
    int numStations,
    float xMin, float xMax,
    float yMin, float yMax,
    float zMin, float zMax,
    float homogeneousVp,
    float T0, float Tf, float alpha,
    int dimension, float initialStepSize,
    int iterPerTemp, int maxIter,
    int coolingSchedule,
    int numChains,
    float* h_bestSolution,
    float* h_misfitHistory)
{
    // Allocate device memory
    StationGPU* d_stations;
    curandState* d_states;
    float* d_solutions;
    float* d_misfitHistory;
    
    size_t stationsSize = numStations * sizeof(StationGPU);
    size_t solutionsSize = numChains * 5 * sizeof(float);
    size_t historySize = numChains * maxIter * sizeof(float);
    
    cudaMalloc(&d_stations, stationsSize);
    cudaMalloc(&d_states, numChains * sizeof(curandState));
    cudaMalloc(&d_solutions, solutionsSize);
    cudaMalloc(&d_misfitHistory, historySize);
    
    // Copy stations to device
    StationGPU* h_stationsGPU = new StationGPU[numStations];
    for (int i = 0; i < numStations; ++i) {
        h_stationsGPU[i].lat = h_stations[i * 3 + 0];
        h_stationsGPU[i].lon = h_stations[i * 3 + 1];
        h_stationsGPU[i].arrivalTime = h_stations[i * 3 + 2];
    }
    cudaMemcpy(d_stations, h_stationsGPU, stationsSize, cudaMemcpyHostToDevice);
    
    // Initialize random states
    int threadsPerBlock = 256;
    int blocksForInit = (numChains + threadsPerBlock - 1) / threadsPerBlock;
    initRandomStates<<<blocksForInit, threadsPerBlock>>>(d_states, time(nullptr), numChains);
    
    // Setup Cauchy parameters
    CauchyParams params;
    params.T0 = T0;
    params.Tf = Tf;
    params.alpha = alpha;
    params.dimension = dimension;
    params.initialStepSize = initialStepSize;
    params.iterPerTemp = iterPerTemp;
    params.maxIter = maxIter;
    params.coolingSchedule = coolingSchedule;
    
    // Launch Cauchy SA kernel
    int blocksPerGrid = (numChains + threadsPerBlock - 1) / threadsPerBlock;
    cauchySAKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_stations, numStations,
        xMin, xMax, yMin, yMax, zMin, zMax,
        homogeneousVp, params,
        d_states, d_solutions, d_misfitHistory,
        numChains
    );
    
    cudaDeviceSynchronize();
    
    // Copy results back
    float* h_allSolutions = new float[numChains * 5];
    cudaMemcpy(h_allSolutions, d_solutions, solutionsSize, cudaMemcpyDeviceToHost);
    
    // Find best among all chains
    float bestMisfit = FLT_MAX;
    int bestChain = 0;
    for (int i = 0; i < numChains; ++i) {
        if (h_allSolutions[i * 5 + 4] < bestMisfit) {
            bestMisfit = h_allSolutions[i * 5 + 4];
            bestChain = i;
        }
    }
    
    // Copy best solution
    memcpy(h_bestSolution, &h_allSolutions[bestChain * 5], 5 * sizeof(float));
    
    // Copy misfit history from best chain
    if (h_misfitHistory != nullptr) {
        cudaMemcpy(h_misfitHistory, &d_misfitHistory[bestChain * maxIter],
                   maxIter * sizeof(float), cudaMemcpyDeviceToHost);
    }
    
    // Cleanup
    cudaFree(d_stations);
    cudaFree(d_states);
    cudaFree(d_solutions);
    cudaFree(d_misfitHistory);
    delete[] h_stationsGPU;
    delete[] h_allSolutions;
}

} // extern "C"
