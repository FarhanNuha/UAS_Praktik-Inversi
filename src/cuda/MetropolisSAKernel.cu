#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cfloat>
#include <cmath>

__constant__ float DEG_TO_RAD = 3.14159265f / 180.0f;
__constant__ float EARTH_RADIUS = 6371.0f;

struct StationGPU {
    float lat;
    float lon;
    float arrivalTime;
};

struct MetropolisParams {
    float T0;
    float Tf;
    float alpha;
    int markovChainLength;
    float targetAcceptance;
    int maxIter;
    int coolingSchedule; // 0=Exp, 1=Linear, 2=Log, 3=Adaptive
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

__device__ float updateTemperatureMetropolis(float T, int iteration, const MetropolisParams& params) {
    switch(params.coolingSchedule) {
        case 0: // Exponential
            return params.T0 * powf(params.alpha, iteration / params.markovChainLength);
        case 1: // Linear
            return params.T0 - params.alpha * iteration;
        case 2: // Logarithmic
            return params.T0 / (1.0f + params.alpha * logf(1.0f + iteration));
        case 3: // Adaptive
            return params.T0 * powf(1.0f - (float)iteration / params.maxIter, params.alpha);
        default:
            return T * params.alpha;
    }
}

__device__ bool metropolisAcceptance(float currentMisfit, float newMisfit, float T, curandState* state) {
    if (newMisfit < currentMisfit) {
        return true;
    }
    
    // Metropolis criterion: P = exp(-ΔE/T)
    float deltaE = newMisfit - currentMisfit;
    float probability = expf(-deltaE / T);
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

__global__ void metropolisSAKernel(
    const StationGPU* stations,
    int numStations,
    float xMin, float xMax,
    float yMin, float yMax,
    float zMin, float zMax,
    float homogeneousVp,
    MetropolisParams params,
    curandState* states,
    float* solutions,
    float* misfitHistory,
    float* acceptanceRates,
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
    int totalAccepted = 0;
    int totalTried = 0;
    
    int markovChainCount = 0;
    
    // Main Metropolis SA loop
    for (int iter = 0; iter < params.maxIter; ++iter) {
        // Markov chain at current temperature
        int acceptedInChain = 0;
        
        for (int mc = 0; mc < params.markovChainLength; ++mc) {
            // Generate neighbor with Gaussian perturbation
            float stepX = (xMax - xMin) * 0.05f * (T / params.T0);
            float stepY = (yMax - yMin) * 0.05f * (T / params.T0);
            float stepZ = (zMax - zMin) * 0.05f * (T / params.T0);
            float stepT0 = 5.0f * (T / params.T0);
            
            // Box-Muller for Gaussian random
            float u1 = curand_uniform(&localState);
            float u2 = curand_uniform(&localState);
            float gaussX = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
            
            u1 = curand_uniform(&localState);
            u2 = curand_uniform(&localState);
            float gaussY = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
            
            u1 = curand_uniform(&localState);
            u2 = curand_uniform(&localState);
            float gaussZ = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
            
            float newX = currentX + gaussX * stepX;
            float newY = currentY + gaussY * stepY;
            float newZ = currentZ + gaussZ * stepZ;
            float newT0 = currentT0 + curand_normal(&localState) * stepT0;
            
            // Apply boundary constraints
            newX = fmaxf(xMin, fminf(xMax, newX));
            newY = fmaxf(yMin, fminf(yMax, newY));
            newZ = fmaxf(zMin, fminf(zMax, newZ));
            
            // Compute new misfit
            float newMisfit = computeMisfitGPU(newX, newY, newZ, newT0,
                                              stations, numStations, homogeneousVp);
            
            // Metropolis acceptance
            if (metropolisAcceptance(currentMisfit, newMisfit, T, &localState)) {
                currentX = newX;
                currentY = newY;
                currentZ = newZ;
                currentT0 = newT0;
                currentMisfit = newMisfit;
                acceptedInChain++;
                
                // Update best
                if (newMisfit < bestMisfit) {
                    bestX = newX;
                    bestY = newY;
                    bestZ = newZ;
                    bestT0 = newT0;
                    bestMisfit = newMisfit;
                }
            }
            
            totalTried++;
        }
        
        totalAccepted += acceptedInChain;
        markovChainCount++;
        
        // Calculate acceptance ratio
        float acceptanceRatio = (float)acceptedInChain / params.markovChainLength;
        
        // Adaptive temperature adjustment based on acceptance ratio
        if (params.coolingSchedule == 3) { // Adaptive
            if (acceptanceRatio > params.targetAcceptance + 0.1f) {
                // Acceptance too high, decrease temperature faster
                T *= 0.85f;
            } else if (acceptanceRatio < params.targetAcceptance - 0.1f) {
                // Acceptance too low, decrease temperature slower
                T *= 0.98f;
            } else {
                T = updateTemperatureMetropolis(T, markovChainCount, params);
            }
        } else {
            T = updateTemperatureMetropolis(T, markovChainCount, params);
        }
        
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
    
    // Store acceptance rate
    acceptanceRates[chainIdx] = (float)totalAccepted / totalTried;
    
    states[chainIdx] = localState;
}

// =============================================================================
// Host Interface
// =============================================================================

extern "C" {

void cudaMetropolisSA(
    const float* h_stations,
    int numStations,
    float xMin, float xMax,
    float yMin, float yMax,
    float zMin, float zMax,
    float homogeneousVp,
    float T0, float Tf, float alpha,
    int markovChainLength, float targetAcceptance,
    int maxIter, int coolingSchedule,
    int numChains,
    float* h_bestSolution,
    float* h_misfitHistory,
    float* h_acceptanceRate)
{
    // Allocate device memory
    StationGPU* d_stations;
    curandState* d_states;
    float* d_solutions;
    float* d_misfitHistory;
    float* d_acceptanceRates;
    
    size_t stationsSize = numStations * sizeof(StationGPU);
    size_t solutionsSize = numChains * 5 * sizeof(float);
    size_t historySize = numChains * maxIter * sizeof(float);
    
    cudaMalloc(&d_stations, stationsSize);
    cudaMalloc(&d_states, numChains * sizeof(curandState));
    cudaMalloc(&d_solutions, solutionsSize);
    cudaMalloc(&d_misfitHistory, historySize);
    cudaMalloc(&d_acceptanceRates, numChains * sizeof(float));
    
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
    
    // Setup Metropolis parameters
    MetropolisParams params;
    params.T0 = T0;
    params.Tf = Tf;
    params.alpha = alpha;
    params.markovChainLength = markovChainLength;
    params.targetAcceptance = targetAcceptance;
    params.maxIter = maxIter;
    params.coolingSchedule = coolingSchedule;
    
    // Launch Metropolis SA kernel
    int blocksPerGrid = (numChains + threadsPerBlock - 1) / threadsPerBlock;
    metropolisSAKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_stations, numStations,
        xMin, xMax, yMin, yMax, zMin, zMax,
        homogeneousVp, params,
        d_states, d_solutions, d_misfitHistory, d_acceptanceRates,
        numChains
    );
    
    cudaDeviceSynchronize();
    
    // Copy results back
    float* h_allSolutions = new float[numChains * 5];
    float* h_allAcceptance = new float[numChains];
    cudaMemcpy(h_allSolutions, d_solutions, solutionsSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_allAcceptance, d_acceptanceRates, numChains * sizeof(float), cudaMemcpyDeviceToHost);
    
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
    
    // Copy acceptance rate
    if (h_acceptanceRate != nullptr) {
        *h_acceptanceRate = h_allAcceptance[bestChain];
    }
    
    // Cleanup
    cudaFree(d_stations);
    cudaFree(d_states);
    cudaFree(d_solutions);
    cudaFree(d_misfitHistory);
    cudaFree(d_acceptanceRates);
    delete[] h_stationsGPU;
    delete[] h_allSolutions;
    delete[] h_allAcceptance;
}

} // extern "C"
