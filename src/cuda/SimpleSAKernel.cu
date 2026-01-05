#include "CUDACommon.cuh"
#include <cfloat>

struct SAParams {
    float T0;
    float Tf;
    float alpha;
    int iterPerTemp;
    int maxIter;
    int coolingSchedule;
};

// =============================================================================
// Device Functions (Specific to Simple SA)
// =============================================================================

__device__ float updateTemperature(float T, int iteration, const SAParams& params) {
    int tempIter = iteration / params.iterPerTemp;
    
    switch(params.coolingSchedule) {
        case 0: return params.T0 * powf(params.alpha, tempIter);
        case 1: return params.T0 - params.alpha * iteration;
        case 2: return params.T0 / (1.0f + params.alpha * logf(1.0f + iteration));
        case 3: return params.T0 / (1.0f + params.alpha * iteration);
        default: return T * params.alpha;
    }
}

__device__ bool acceptSolution(float currentMisfit, float newMisfit, float T, curandState* state) {
    if (newMisfit < currentMisfit) return true;
    float delta = newMisfit - currentMisfit;
    float probability = expf(-delta / T);
    return curand_uniform(state) < probability;
}

// =============================================================================
// Kernels
// =============================================================================

__global__ void simpleSAKernel(
    const StationGPU* stations, int numStations,
    float xMin, float xMax, float yMin, float yMax, float zMin, float zMax,
    float homogeneousVp, SAParams params,
    curandState* states, float* solutions, float* misfitHistory, int numChains)
{
    int chainIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (chainIdx >= numChains) return;
    
    curandState localState = states[chainIdx];
    
    float currentX = xMin + curand_uniform(&localState) * (xMax - xMin);
    float currentY = yMin + curand_uniform(&localState) * (yMax - yMin);
    float currentZ = zMin + curand_uniform(&localState) * (zMax - zMin);
    float currentT0 = curand_uniform(&localState) * 100.0f - 50.0f;
    
    float currentMisfit = computeMisfitDevice(currentX, currentY, currentZ, currentT0, 
                                           stations, numStations, homogeneousVp);
    
    float bestX = currentX, bestY = currentY, bestZ = currentZ, bestT0 = currentT0;
    float bestMisfit = currentMisfit;
    float T = params.T0;
    
    for (int iter = 0; iter < params.maxIter; ++iter) {
        float stepX = (xMax - xMin) * 0.1f * (T / params.T0);
        float stepY = (yMax - yMin) * 0.1f * (T / params.T0);
        float stepZ = (zMax - zMin) * 0.1f * (T / params.T0);
        float stepT0 = 10.0f * (T / params.T0);
        
        float newX = currentX + (curand_uniform(&localState) * 2.0f - 1.0f) * stepX;
        float newY = currentY + (curand_uniform(&localState) * 2.0f - 1.0f) * stepY;
        float newZ = currentZ + (curand_uniform(&localState) * 2.0f - 1.0f) * stepZ;
        float newT0 = currentT0 + (curand_uniform(&localState) * 2.0f - 1.0f) * stepT0;
        
        newX = fmaxf(xMin, fminf(xMax, newX));
        newY = fmaxf(yMin, fminf(yMax, newY));
        newZ = fmaxf(zMin, fminf(zMax, newZ));
        
        float newMisfit = computeMisfitDevice(newX, newY, newZ, newT0,
                                          stations, numStations, homogeneousVp);
        
        if (acceptSolution(currentMisfit, newMisfit, T, &localState)) {
            currentX = newX; currentY = newY; currentZ = newZ; currentT0 = newT0;
            currentMisfit = newMisfit;
            
            if (newMisfit < bestMisfit) {
                bestX = newX; bestY = newY; bestZ = newZ; bestT0 = newT0;
                bestMisfit = newMisfit;
            }
        }
        
        if (iter % params.iterPerTemp == 0) T = updateTemperature(T, iter, params);
        misfitHistory[chainIdx * params.maxIter + iter] = bestMisfit;
        if (T < params.Tf) break;
    }
    
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

void cudaSimpleSA(
    const float* h_stations, int numStations,
    float xMin, float xMax, float yMin, float yMax, float zMin, float zMax,
    float homogeneousVp,
    float T0, float Tf, float alpha, int iterPerTemp, int maxIter, int coolingSchedule,
    int numChains, float* h_bestSolution, float* h_misfitHistory)
{
    StationGPU* d_stations;
    curandState* d_states;
    float *d_solutions, *d_misfitHistory;
    
    size_t stationsSize = numStations * sizeof(StationGPU);
    size_t solutionsSize = numChains * 5 * sizeof(float);
    size_t historySize = numChains * maxIter * sizeof(float);
    
    cudaMalloc(&d_stations, stationsSize);
    cudaMalloc(&d_states, numChains * sizeof(curandState));
    cudaMalloc(&d_solutions, solutionsSize);
    cudaMalloc(&d_misfitHistory, historySize);
    
    StationGPU* h_stationsGPU = new StationGPU[numStations];
    for (int i = 0; i < numStations; ++i) {
        h_stationsGPU[i].lat = h_stations[i * 3 + 0];
        h_stationsGPU[i].lon = h_stations[i * 3 + 1];
        h_stationsGPU[i].arrivalTime = h_stations[i * 3 + 2];
    }
    cudaMemcpy(d_stations, h_stationsGPU, stationsSize, cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksForInit = (numChains + threadsPerBlock - 1) / threadsPerBlock;
    initRandomStatesKernel<<<blocksForInit, threadsPerBlock>>>(d_states, time(nullptr), numChains);
    
    SAParams params = {T0, Tf, alpha, iterPerTemp, maxIter, coolingSchedule};
    
    int blocksPerGrid = (numChains + threadsPerBlock - 1) / threadsPerBlock;
    simpleSAKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_stations, numStations, xMin, xMax, yMin, yMax, zMin, zMax,
        homogeneousVp, params, d_states, d_solutions, d_misfitHistory, numChains
    );
    
    cudaDeviceSynchronize();
    
    float* h_allSolutions = new float[numChains * 5];
    cudaMemcpy(h_allSolutions, d_solutions, solutionsSize, cudaMemcpyDeviceToHost);
    
    float bestMisfit = FLT_MAX;
    int bestChain = 0;
    for (int i = 0; i < numChains; ++i) {
        if (h_allSolutions[i * 5 + 4] < bestMisfit) {
            bestMisfit = h_allSolutions[i * 5 + 4];
            bestChain = i;
        }
    }
    
    memcpy(h_bestSolution, &h_allSolutions[bestChain * 5], 5 * sizeof(float));
    
    if (h_misfitHistory != nullptr) {
        cudaMemcpy(h_misfitHistory, &d_misfitHistory[bestChain * maxIter], 
                   maxIter * sizeof(float), cudaMemcpyDeviceToHost);
    }
    
    cudaFree(d_stations);
    cudaFree(d_states);
    cudaFree(d_solutions);
    cudaFree(d_misfitHistory);
    delete[] h_stationsGPU;
    delete[] h_allSolutions;
}

} // extern "C"
