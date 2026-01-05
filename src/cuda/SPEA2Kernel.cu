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

struct IndividualSPEA2 {
    float x, y, z, t0;
    float fitness;           // Raw fitness (misfit)
    float strength;          // Strength value
    float rawFitness;        // R(i) = sum of strengths
    float density;           // D(i) = 1/(σk + 2)
    float spea2Fitness;      // F(i) = R(i) + D(i)
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

__device__ float euclideanDistance(const IndividualSPEA2& a, const IndividualSPEA2& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    float dt = a.t0 - b.t0;
    return sqrtf(dx*dx + dy*dy + dz*dz + dt*dt);
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
    IndividualSPEA2* population,
    float xMin, float xMax,
    float yMin, float yMax,
    float zMin, float zMax,
    curandState* states,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    curandState localState = states[idx];
    
    population[idx].x = xMin + curand_uniform(&localState) * (xMax - xMin);
    population[idx].y = yMin + curand_uniform(&localState) * (yMax - yMin);
    population[idx].z = zMin + curand_uniform(&localState) * (zMax - zMin);
    population[idx].t0 = curand_uniform(&localState) * 100.0f - 50.0f;
    population[idx].fitness = FLT_MAX;
    population[idx].strength = 0.0f;
    population[idx].rawFitness = 0.0f;
    population[idx].density = 0.0f;
    population[idx].spea2Fitness = FLT_MAX;
    
    states[idx] = localState;
}

__global__ void evaluateFitness(
    IndividualSPEA2* population,
    const StationGPU* stations,
    int numStations,
    float homogeneousVp,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    IndividualSPEA2& ind = population[idx];
    ind.fitness = computeFitnessGPU(ind.x, ind.y, ind.z, ind.t0,
                                    stations, numStations, homogeneousVp);
}

// Calculate strength: number of individuals dominated
__global__ void calculateStrength(
    IndividualSPEA2* population,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    int dominated = 0;
    float myFitness = population[idx].fitness;
    
    for (int j = 0; j < size; ++j) {
        if (j != idx && myFitness < population[j].fitness) {
            dominated++;
        }
    }
    
    population[idx].strength = (float)dominated;
}

// Calculate raw fitness: sum of strengths of dominators
__global__ void calculateRawFitness(
    IndividualSPEA2* population,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float rawFit = 0.0f;
    float myFitness = population[idx].fitness;
    
    for (int j = 0; j < size; ++j) {
        if (j != idx && population[j].fitness < myFitness) {
            rawFit += population[j].strength;
        }
    }
    
    population[idx].rawFitness = rawFit;
}

// Calculate density using k-th nearest neighbor
__global__ void calculateDensity(
    IndividualSPEA2* population,
    float* distances,
    int size,
    int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // Calculate distances to all other individuals
    for (int j = 0; j < size; ++j) {
        if (j == idx) {
            distances[idx * size + j] = FLT_MAX;
        } else {
            distances[idx * size + j] = euclideanDistance(population[idx], population[j]);
        }
    }
    
    __syncthreads();
    
    // Simple selection sort to find k-th nearest (inefficient but simple)
    float kthDistance = FLT_MAX;
    int count = 0;
    
    for (int iter = 0; iter < size && count <= k; ++iter) {
        float minDist = FLT_MAX;
        for (int j = 0; j < size; ++j) {
            if (distances[idx * size + j] < minDist) {
                minDist = distances[idx * size + j];
            }
        }
        
        if (minDist < FLT_MAX) {
            count++;
            if (count == k) {
                kthDistance = minDist;
            }
            
            // Mark as used
            for (int j = 0; j < size; ++j) {
                if (distances[idx * size + j] == minDist) {
                    distances[idx * size + j] = FLT_MAX;
                    break;
                }
            }
        }
    }
    
    // Density = 1 / (σk + 2)
    population[idx].density = 1.0f / (kthDistance + 2.0f);
    population[idx].spea2Fitness = population[idx].rawFitness + population[idx].density;
}

__global__ void crossoverAndMutation(
    const IndividualSPEA2* parents,
    IndividualSPEA2* offspring,
    float xMin, float xMax,
    float yMin, float yMax,
    float zMin, float zMax,
    float crossoverRate,
    float mutationRate,
    curandState* states,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size / 2) return;
    
    int parent1Idx = idx * 2;
    int parent2Idx = idx * 2 + 1;
    
    if (parent2Idx >= size) return;
    
    curandState localState = states[idx];
    
    IndividualSPEA2 child1 = parents[parent1Idx];
    IndividualSPEA2 child2 = parents[parent2Idx];
    
    // SBX Crossover
    if (curand_uniform(&localState) < crossoverRate) {
        float eta = 2.0f;
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
    
    // Mutation
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
    
    // Boundary constraints
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

// =============================================================================
// Host Interface
// =============================================================================

extern "C" {

void cudaSPEA2(
    const float* h_stations,
    int numStations,
    float xMin, float xMax,
    float yMin, float yMax,
    float zMin, float zMax,
    float homogeneousVp,
    int populationSize,
    int archiveSize,
    int maxGenerations,
    float crossoverRate,
    float mutationRate,
    int kNearest,
    float* h_bestSolution,
    float* h_generationBest)
{
    // Implementation simplified for single objective
    // Full SPEA2 would track Pareto front
    
    // For this implementation, we use SPEA2 fitness but optimize single objective
    
    // Allocate device memory
    StationGPU* d_stations;
    IndividualSPEA2* d_population;
    IndividualSPEA2* d_archive;
    IndividualSPEA2* d_combined;
    IndividualSPEA2* d_offspring;
    float* d_distances;
    curandState* d_states;
    
    size_t stationsSize = numStations * sizeof(StationGPU);
    size_t popSize = populationSize * sizeof(IndividualSPEA2);
    size_t archSize = archiveSize * sizeof(IndividualSPEA2);
    size_t combSize = (populationSize + archiveSize) * sizeof(IndividualSPEA2);
    size_t distSize = (populationSize + archiveSize) * (populationSize + archiveSize) * sizeof(float);
    
    cudaMalloc(&d_stations, stationsSize);
    cudaMalloc(&d_population, popSize);
    cudaMalloc(&d_archive, archSize);
    cudaMalloc(&d_combined, combSize);
    cudaMalloc(&d_offspring, popSize);
    cudaMalloc(&d_distances, distSize);
    cudaMalloc(&d_states, populationSize * sizeof(curandState));
    
    // Copy stations
    StationGPU* h_stationsGPU = new StationGPU[numStations];
    for (int i = 0; i < numStations; ++i) {
        h_stationsGPU[i].lat = h_stations[i * 3 + 0];
        h_stationsGPU[i].lon = h_stations[i * 3 + 1];
        h_stationsGPU[i].arrivalTime = h_stations[i * 3 + 2];
    }
    cudaMemcpy(d_stations, h_stationsGPU, stationsSize, cudaMemcpyHostToDevice);
    
    // Initialize
    int threadsPerBlock = 256;
    int blocksForPop = (populationSize + threadsPerBlock - 1) / threadsPerBlock;
    initRandomStates<<<blocksForPop, threadsPerBlock>>>(d_states, time(nullptr), populationSize);
    
    initializePopulation<<<blocksForPop, threadsPerBlock>>>(
        d_population, xMin, xMax, yMin, yMax, zMin, zMax, d_states, populationSize
    );
    
    // Initialize empty archive
    initializePopulation<<<(archiveSize + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
        d_archive, xMin, xMax, yMin, yMax, zMin, zMax, d_states, archiveSize
    );
    
    evaluateFitness<<<blocksForPop, threadsPerBlock>>>(
        d_population, d_stations, numStations, homogeneousVp, populationSize
    );
    
    cudaDeviceSynchronize();
    
    // Main SPEA2 loop (simplified)
    IndividualSPEA2* h_best = new IndividualSPEA2;
    h_best->fitness = FLT_MAX;
    
    for (int gen = 0; gen < maxGenerations; ++gen) {
        // Evaluate archive if not first generation
        if (gen > 0) {
            evaluateFitness<<<(archiveSize + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
                d_archive, d_stations, numStations, homogeneousVp, archiveSize
            );
        }
        
        // Generate offspring from archive (or population in first gen)
        IndividualSPEA2* parents = (gen == 0) ? d_population : d_archive;
        int parentsSize = (gen == 0) ? populationSize : archiveSize;
        
        crossoverAndMutation<<<(populationSize / 2 + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
            parents, d_offspring,
            xMin, xMax, yMin, yMax, zMin, zMax,
            crossoverRate, mutationRate, d_states, populationSize
        );
        
        evaluateFitness<<<blocksForPop, threadsPerBlock>>>(
            d_offspring, d_stations, numStations, homogeneousVp, populationSize
        );
        
        // Simple archive update: keep best individuals
        // (Full SPEA2 would use environmental selection)
        cudaMemcpy(d_population, d_offspring, popSize, cudaMemcpyDeviceToDevice);
        
        // Update archive with best from population
        // Simplified: just sort and take best
        // TODO: Implement proper SPEA2 environmental selection
        
        cudaDeviceSynchronize();
        
        // Track best
        IndividualSPEA2* h_pop = new IndividualSPEA2[populationSize];
        cudaMemcpy(h_pop, d_population, popSize, cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < populationSize; ++i) {
            if (h_pop[i].fitness < h_best->fitness) {
                *h_best = h_pop[i];
            }
        }
        
        if (h_generationBest != nullptr) {
            h_generationBest[gen] = h_best->fitness;
        }
        
        delete[] h_pop;
    }
    
    // Return best solution
    h_bestSolution[0] = h_best->x;
    h_bestSolution[1] = h_best->y;
    h_bestSolution[2] = h_best->z;
    h_bestSolution[3] = h_best->t0;
    h_bestSolution[4] = h_best->fitness;
    
    // Cleanup
    cudaFree(d_stations);
    cudaFree(d_population);
    cudaFree(d_archive);
    cudaFree(d_combined);
    cudaFree(d_offspring);
    cudaFree(d_distances);
    cudaFree(d_states);
    delete[] h_stationsGPU;
    delete h_best;
}

} // extern "C"
