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

struct IndividualGPU {
    float x, y, z, t0;
    float fitness;
};

struct GAParams {
    int populationSize;
    int maxGenerations;
    float crossoverRate;
    float mutationRate;
    int eliteSize;
    int tournamentSize;
    bool realCoded;
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

__device__ float computeFitnessGPU(
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

// =============================================================================
// Kernels
// =============================================================================

__global__ void initRandomStates(curandState* states, unsigned long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void initializePopulation(
    IndividualGPU* population,
    float xMin, float xMax,
    float yMin, float yMax,
    float zMin, float zMax,
    curandState* states,
    int populationSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= populationSize) return;
    
    curandState localState = states[idx];
    
    population[idx].x = xMin + curand_uniform(&localState) * (xMax - xMin);
    population[idx].y = yMin + curand_uniform(&localState) * (yMax - yMin);
    population[idx].z = zMin + curand_uniform(&localState) * (zMax - zMin);
    population[idx].t0 = curand_uniform(&localState) * 100.0f - 50.0f;
    population[idx].fitness = FLT_MAX;
    
    states[idx] = localState;
}

__global__ void evaluateFitness(
    IndividualGPU* population,
    const StationGPU* stations,
    int numStations,
    float homogeneousVp,
    int populationSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= populationSize) return;
    
    IndividualGPU& ind = population[idx];
    ind.fitness = computeFitnessGPU(ind.x, ind.y, ind.z, ind.t0,
                                    stations, numStations, homogeneousVp);
}

__global__ void tournamentSelection(
    const IndividualGPU* population,
    IndividualGPU* parents,
    curandState* states,
    int populationSize,
    int tournamentSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= populationSize) return;
    
    curandState localState = states[idx];
    
    // Tournament selection
    int bestIdx = (int)(curand_uniform(&localState) * populationSize);
    float bestFitness = population[bestIdx].fitness;
    
    for (int i = 1; i < tournamentSize; ++i) {
        int candidateIdx = (int)(curand_uniform(&localState) * populationSize);
        if (population[candidateIdx].fitness < bestFitness) {
            bestIdx = candidateIdx;
            bestFitness = population[candidateIdx].fitness;
        }
    }
    
    parents[idx] = population[bestIdx];
    states[idx] = localState;
}

__global__ void crossoverAndMutation(
    const IndividualGPU* parents,
    IndividualGPU* offspring,
    float xMin, float xMax,
    float yMin, float yMax,
    float zMin, float zMax,
    float crossoverRate,
    float mutationRate,
    curandState* states,
    int populationSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= populationSize / 2) return;
    
    int parent1Idx = idx * 2;
    int parent2Idx = idx * 2 + 1;
    
    if (parent2Idx >= populationSize) return;
    
    curandState localState = states[idx];
    
    IndividualGPU child1 = parents[parent1Idx];
    IndividualGPU child2 = parents[parent2Idx];
    
    // Simulated Binary Crossover (SBX)
    if (curand_uniform(&localState) < crossoverRate) {
        float eta = 2.0f; // Distribution index
        float u = curand_uniform(&localState);
        float beta = (u <= 0.5f) ? powf(2*u, 1.0f/(eta+1)) : powf(1.0f/(2*(1-u)), 1.0f/(eta+1));
        
        float p1x = parents[parent1Idx].x;
        float p2x = parents[parent2Idx].x;
        child1.x = 0.5f * ((1 + beta) * p1x + (1 - beta) * p2x);
        child2.x = 0.5f * ((1 - beta) * p1x + (1 + beta) * p2x);
        
        float p1y = parents[parent1Idx].y;
        float p2y = parents[parent2Idx].y;
        child1.y = 0.5f * ((1 + beta) * p1y + (1 - beta) * p2y);
        child2.y = 0.5f * ((1 - beta) * p1y + (1 + beta) * p2y);
        
        float p1z = parents[parent1Idx].z;
        float p2z = parents[parent2Idx].z;
        child1.z = 0.5f * ((1 + beta) * p1z + (1 - beta) * p2z);
        child2.z = 0.5f * ((1 - beta) * p1z + (1 + beta) * p2z);
        
        float p1t0 = parents[parent1Idx].t0;
        float p2t0 = parents[parent2Idx].t0;
        child1.t0 = 0.5f * ((1 + beta) * p1t0 + (1 - beta) * p2t0);
        child2.t0 = 0.5f * ((1 - beta) * p1t0 + (1 + beta) * p2t0);
    }
    
    // Gaussian Mutation
    if (curand_uniform(&localState) < mutationRate) {
        child1.x += curand_normal(&localState) * (xMax - xMin) * 0.1f;
        child1.y += curand_normal(&localState) * (yMax - yMin) * 0.1f;
        child1.z += curand_normal(&localState) * (zMax - zMin) * 0.1f;
        child1.t0 += curand_normal(&localState) * 10.0f;
    }
    
    if (curand_uniform(&localState) < mutationRate) {
        child2.x += curand_normal(&localState) * (xMax - xMin) * 0.1f;
        child2.y += curand_normal(&localState) * (yMax - yMin) * 0.1f;
        child2.z += curand_normal(&localState) * (zMax - zMin) * 0.1f;
        child2.t0 += curand_normal(&localState) * 10.0f;
    }
    
    // Apply boundary constraints
    child1.x = fmaxf(xMin, fminf(xMax, child1.x));
    child1.y = fmaxf(yMin, fminf(yMax, child1.y));
    child1.z = fmaxf(zMin, fminf(zMax, child1.z));
    
    child2.x = fmaxf(xMin, fminf(xMax, child2.x));
    child2.y = fmaxf(yMin, fminf(yMax, child2.y));
    child2.z = fmaxf(zMin, fminf(zMax, child2.z));
    
    offspring[parent1Idx] = child1;
    offspring[parent2Idx] = child2;
    
    states[idx] = localState;
}

// Simple sort for small populations (elite selection)
__global__ void sortPopulationByFitness(IndividualGPU* population, int populationSize) {
    // Bubble sort (simple but inefficient - use for small elite sizes only)
    for (int i = 0; i < populationSize - 1; ++i) {
        for (int j = 0; j < populationSize - i - 1; ++j) {
            if (population[j].fitness > population[j+1].fitness) {
                IndividualGPU temp = population[j];
                population[j] = population[j+1];
                population[j+1] = temp;
            }
        }
    }
}

// =============================================================================
// Host Interface
// =============================================================================

extern "C" {

void cudaStandardGA(
    const float* h_stations,
    int numStations,
    float xMin, float xMax,
    float yMin, float yMax,
    float zMin, float zMax,
    float homogeneousVp,
    int populationSize,
    int maxGenerations,
    float crossoverRate,
    float mutationRate,
    int eliteSize,
    int tournamentSize,
    float* h_bestSolution,
    float* h_generationBest)
{
    // Allocate device memory
    StationGPU* d_stations;
    IndividualGPU* d_population;
    IndividualGPU* d_offspring;
    IndividualGPU* d_parents;
    curandState* d_states;
    
    size_t stationsSize = numStations * sizeof(StationGPU);
    size_t populationSize_bytes = populationSize * sizeof(IndividualGPU);
    
    cudaMalloc(&d_stations, stationsSize);
    cudaMalloc(&d_population, populationSize_bytes);
    cudaMalloc(&d_offspring, populationSize_bytes);
    cudaMalloc(&d_parents, populationSize_bytes);
    cudaMalloc(&d_states, populationSize * sizeof(curandState));
    
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
    int blocksPerGrid = (populationSize + threadsPerBlock - 1) / threadsPerBlock;
    initRandomStates<<<blocksPerGrid, threadsPerBlock>>>(d_states, time(nullptr), populationSize);
    
    // Initialize population
    initializePopulation<<<blocksPerGrid, threadsPerBlock>>>(
        d_population, xMin, xMax, yMin, yMax, zMin, zMax, d_states, populationSize
    );
    
    // Evaluate initial population
    evaluateFitness<<<blocksPerGrid, threadsPerBlock>>>(
        d_population, d_stations, numStations, homogeneousVp, populationSize
    );
    
    cudaDeviceSynchronize();
    
    // Main GA loop
    IndividualGPU* h_population = new IndividualGPU[populationSize];
    
    for (int gen = 0; gen < maxGenerations; ++gen) {
        // Tournament Selection
        tournamentSelection<<<blocksPerGrid, threadsPerBlock>>>(
            d_population, d_parents, d_states, populationSize, tournamentSize
        );
        
        // Crossover and Mutation
        int blocksForCrossover = (populationSize / 2 + threadsPerBlock - 1) / threadsPerBlock;
        crossoverAndMutation<<<blocksForCrossover, threadsPerBlock>>>(
            d_parents, d_offspring,
            xMin, xMax, yMin, yMax, zMin, zMax,
            crossoverRate, mutationRate,
            d_states, populationSize
        );
        
        // Evaluate offspring
        evaluateFitness<<<blocksPerGrid, threadsPerBlock>>>(
            d_offspring, d_stations, numStations, homogeneousVp, populationSize
        );
        
        cudaDeviceSynchronize();
        
        // Elitism: copy best individuals from old population
        // First, sort population to get elite
        sortPopulationByFitness<<<1, 1>>>(d_population, populationSize);
        cudaDeviceSynchronize();
        
        // Copy elite to offspring (replace worst)
        cudaMemcpy(d_offspring, d_population, eliteSize * sizeof(IndividualGPU), cudaMemcpyDeviceToDevice);
        
        // Replace population with offspring
        cudaMemcpy(d_population, d_offspring, populationSize_bytes, cudaMemcpyDeviceToDevice);
        
        // Track best fitness for this generation
        if (h_generationBest != nullptr) {
            cudaMemcpy(h_population, d_population, sizeof(IndividualGPU), cudaMemcpyDeviceToHost);
            h_generationBest[gen] = h_population[0].fitness;
        }
    }
    
    // Get final best solution
    sortPopulationByFitness<<<1, 1>>>(d_population, populationSize);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_population, d_population, sizeof(IndividualGPU), cudaMemcpyDeviceToHost);
    
    h_bestSolution[0] = h_population[0].x;
    h_bestSolution[1] = h_population[0].y;
    h_bestSolution[2] = h_population[0].z;
    h_bestSolution[3] = h_population[0].t0;
    h_bestSolution[4] = h_population[0].fitness;
    
    // Cleanup
    cudaFree(d_stations);
    cudaFree(d_population);
    cudaFree(d_offspring);
    cudaFree(d_parents);
    cudaFree(d_states);
    delete[] h_stationsGPU;
    delete[] h_population;
}

} // extern "C"
