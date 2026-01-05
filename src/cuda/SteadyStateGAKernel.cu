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

// Steady State: Each thread generates one offspring
__global__ void steadyStateIteration(
    IndividualGPU* population,
    const StationGPU* stations,
    int numStations,
    float homogeneousVp,
    float xMin, float xMax,
    float yMin, float yMax,
    float zMin, float zMax,
    float crossoverRate,
    float mutationRate,
    int tournamentSize,
    curandState* states,
    int populationSize,
    int numOffspring)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numOffspring) return;
    
    curandState localState = states[idx];
    
    // Tournament selection for parent 1
    int parent1Idx = (int)(curand_uniform(&localState) * populationSize);
    float parent1Fitness = population[parent1Idx].fitness;
    for (int i = 1; i < tournamentSize; ++i) {
        int candidateIdx = (int)(curand_uniform(&localState) * populationSize);
        if (population[candidateIdx].fitness < parent1Fitness) {
            parent1Idx = candidateIdx;
            parent1Fitness = population[candidateIdx].fitness;
        }
    }
    
    // Tournament selection for parent 2
    int parent2Idx = (int)(curand_uniform(&localState) * populationSize);
    float parent2Fitness = population[parent2Idx].fitness;
    for (int i = 1; i < tournamentSize; ++i) {
        int candidateIdx = (int)(curand_uniform(&localState) * populationSize);
        if (population[candidateIdx].fitness < parent2Fitness) {
            parent2Idx = candidateIdx;
            parent2Fitness = population[candidateIdx].fitness;
        }
    }
    
    IndividualGPU offspring;
    
    // Crossover
    if (curand_uniform(&localState) < crossoverRate) {
        // SBX crossover
        float eta = 2.0f;
        float u = curand_uniform(&localState);
        float beta = (u <= 0.5f) ? powf(2*u, 1.0f/(eta+1)) : powf(1.0f/(2*(1-u)), 1.0f/(eta+1));
        
        offspring.x = 0.5f * ((1 + beta) * population[parent1Idx].x + (1 - beta) * population[parent2Idx].x);
        offspring.y = 0.5f * ((1 + beta) * population[parent1Idx].y + (1 - beta) * population[parent2Idx].y);
        offspring.z = 0.5f * ((1 + beta) * population[parent1Idx].z + (1 - beta) * population[parent2Idx].z);
        offspring.t0 = 0.5f * ((1 + beta) * population[parent1Idx].t0 + (1 - beta) * population[parent2Idx].t0);
    } else {
        // Copy parent 1
        offspring = population[parent1Idx];
    }
    
    // Mutation
    if (curand_uniform(&localState) < mutationRate) {
        offspring.x += curand_normal(&localState) * (xMax - xMin) * 0.1f;
        offspring.y += curand_normal(&localState) * (yMax - yMin) * 0.1f;
        offspring.z += curand_normal(&localState) * (zMax - zMin) * 0.1f;
        offspring.t0 += curand_normal(&localState) * 10.0f;
        
        // Apply boundary constraints
        offspring.x = fmaxf(xMin, fminf(xMax, offspring.x));
        offspring.y = fmaxf(yMin, fminf(yMax, offspring.y));
        offspring.z = fmaxf(zMin, fminf(zMax, offspring.z));
    }
    
    // Evaluate offspring
    offspring.fitness = computeFitnessGPU(offspring.x, offspring.y, offspring.z, offspring.t0,
                                         stations, numStations, homogeneousVp);
    
    // Replacement strategy: Replace worst
    // Find worst individual in population
    int worstIdx = 0;
    float worstFitness = population[0].fitness;
    for (int i = 1; i < populationSize; ++i) {
        if (population[i].fitness > worstFitness) {
            worstIdx = i;
            worstFitness = population[i].fitness;
        }
    }
    
    // Replace worst if offspring is better
    if (offspring.fitness < worstFitness) {
        // Use atomic operation to ensure thread safety
        unsigned int* worst_fitness_uint = (unsigned int*)&population[worstIdx].fitness;
        unsigned int old_uint = atomicCAS(worst_fitness_uint, 
                                         __float_as_uint(worstFitness),
                                         __float_as_uint(offspring.fitness));
        
        if (old_uint == __float_as_uint(worstFitness)) {
            // Successfully updated, copy offspring
            population[worstIdx] = offspring;
        }
    }
    
    states[idx] = localState;
}

// Find best individual
__global__ void findBestIndividual(
    const IndividualGPU* population,
    IndividualGPU* bestIndividual,
    int populationSize)
{
    __shared__ IndividualGPU sharedBest[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread loads one individual
    if (idx < populationSize) {
        sharedBest[tid] = population[idx];
    } else {
        sharedBest[tid].fitness = FLT_MAX;
    }
    
    __syncthreads();
    
    // Reduction to find best in block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sharedBest[tid + stride].fitness < sharedBest[tid].fitness) {
                sharedBest[tid] = sharedBest[tid + stride];
            }
        }
        __syncthreads();
    }
    
    // First thread writes block result
    if (tid == 0) {
        atomicMin((int*)&bestIndividual->fitness, __float_as_int(sharedBest[0].fitness));
        if (sharedBest[0].fitness == bestIndividual->fitness) {
            *bestIndividual = sharedBest[0];
        }
    }
}

// =============================================================================
// Host Interface
// =============================================================================

extern "C" {

void cudaSteadyStateGA(
    const float* h_stations,
    int numStations,
    float xMin, float xMax,
    float yMin, float yMax,
    float zMin, float zMax,
    float homogeneousVp,
    int populationSize,
    int maxIterations,
    int offspringPerIter,
    float crossoverRate,
    float mutationRate,
    int tournamentSize,
    float* h_bestSolution,
    float* h_iterationBest)
{
    // Allocate device memory
    StationGPU* d_stations;
    IndividualGPU* d_population;
    IndividualGPU* d_bestIndividual;
    curandState* d_states;
    
    size_t stationsSize = numStations * sizeof(StationGPU);
    size_t populationSize_bytes = populationSize * sizeof(IndividualGPU);
    
    cudaMalloc(&d_stations, stationsSize);
    cudaMalloc(&d_population, populationSize_bytes);
    cudaMalloc(&d_bestIndividual, sizeof(IndividualGPU));
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
    
    // Initialize best individual
    IndividualGPU h_best;
    h_best.fitness = FLT_MAX;
    cudaMemcpy(d_bestIndividual, &h_best, sizeof(IndividualGPU), cudaMemcpyHostToDevice);
    
    // Main Steady State GA loop
    int blocksForOffspring = (offspringPerIter + threadsPerBlock - 1) / threadsPerBlock;
    
    for (int iter = 0; iter < maxIterations; ++iter) {
        // Generate and evaluate offspring, replace worst
        steadyStateIteration<<<blocksForOffspring, threadsPerBlock>>>(
            d_population, d_stations, numStations, homogeneousVp,
            xMin, xMax, yMin, yMax, zMin, zMax,
            crossoverRate, mutationRate, tournamentSize,
            d_states, populationSize, offspringPerIter
        );
        
        cudaDeviceSynchronize();
        
        // Track best every N iterations
        if (h_iterationBest != nullptr && iter % 100 == 0) {
            findBestIndividual<<<blocksPerGrid, threadsPerBlock>>>(
                d_population, d_bestIndividual, populationSize
            );
            cudaDeviceSynchronize();
            
            cudaMemcpy(&h_best, d_bestIndividual, sizeof(IndividualGPU), cudaMemcpyDeviceToHost);
            h_iterationBest[iter / 100] = h_best.fitness;
        }
    }
    
    // Get final best solution
    findBestIndividual<<<blocksPerGrid, threadsPerBlock>>>(
        d_population, d_bestIndividual, populationSize
    );
    cudaDeviceSynchronize();
    
    cudaMemcpy(&h_best, d_bestIndividual, sizeof(IndividualGPU), cudaMemcpyDeviceToHost);
    
    h_bestSolution[0] = h_best.x;
    h_bestSolution[1] = h_best.y;
    h_bestSolution[2] = h_best.z;
    h_bestSolution[3] = h_best.t0;
    h_bestSolution[4] = h_best.fitness;
    
    // Cleanup
    cudaFree(d_stations);
    cudaFree(d_population);
    cudaFree(d_bestIndividual);
    cudaFree(d_states);
    delete[] h_stationsGPU;
}

} // extern "C"
