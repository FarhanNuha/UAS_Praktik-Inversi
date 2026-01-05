#ifndef CUDA_COMMON_CUH
#define CUDA_COMMON_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>

// Use static inline to avoid multiple definition
// Each .cu file gets its own copy (no linking)

namespace CUDAConstants {
    static constexpr float DEG_TO_RAD = 3.14159265f / 180.0f;
    static constexpr float EARTH_RADIUS = 6371.0f;
    static constexpr float PI = 3.14159265f;
}

struct StationGPU {
    float lat;
    float lon;
    float arrivalTime;
};

struct IndividualGPU {
    float x, y, z, t0;
    float fitness;
};

struct IndividualSPEA2 {
    float x, y, z, t0;
    float fitness;
    float strength;
    float rawFitness;
    float density;
    float spea2Fitness;
};

// Static device functions - each translation unit gets its own copy
static __device__ __forceinline__ float haversineDistanceDevice(float lat1, float lon1, float lat2, float lon2) {
    using namespace CUDAConstants;
    float dLat = (lat2 - lat1) * DEG_TO_RAD;
    float dLon = (lon2 - lon1) * DEG_TO_RAD;
    lat1 *= DEG_TO_RAD;
    lat2 *= DEG_TO_RAD;
    float a = sinf(dLat/2) * sinf(dLat/2) + cosf(lat1) * cosf(lat2) * sinf(dLon/2) * sinf(dLon/2);
    float c = 2 * atan2f(sqrtf(a), sqrtf(1-a));
    return EARTH_RADIUS * c;
}

static __device__ __forceinline__ float computeMisfitDevice(
    float x, float y, float z, float t0,
    const StationGPU* stations, int numStations, float homogeneousVp)
{
    float misfit = 0.0f;
    for (int i = 0; i < numStations; ++i) {
        float epicentralDist = haversineDistanceDevice(y, x, stations[i].lat, stations[i].lon);
        float distance3D = sqrtf(epicentralDist * epicentralDist + z * z);
        float travelTime = distance3D / homogeneousVp;
        float predictedArrival = t0 + travelTime;
        float residual = stations[i].arrivalTime - predictedArrival;
        misfit += residual * residual;
    }
    return sqrtf(misfit / numStations);
}

static __device__ __forceinline__ float computeFitnessDevice(
    float x, float y, float z, float t0,
    const StationGPU* stations, int numStations, float homogeneousVp)
{
    return computeMisfitDevice(x, y, z, t0, stations, numStations, homogeneousVp);
}

static __device__ __forceinline__ float cauchyRandomDevice(curandState* state) {
    using namespace CUDAConstants;
    float u = curand_uniform(state);
    return tanf(PI * (u - 0.5f));
}

static __device__ __forceinline__ float euclideanDistanceDevice(const IndividualSPEA2& a, const IndividualSPEA2& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    float dt = a.t0 - b.t0;
    return sqrtf(dx*dx + dy*dy + dz*dz + dt*dt);
}

// Static global function - each file gets its own copy
static __global__ void initRandomStatesKernel(curandState* states, unsigned long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

#endif // CUDA_COMMON_CUH
