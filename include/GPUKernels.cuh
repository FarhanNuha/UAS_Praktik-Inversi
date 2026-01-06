#ifndef GPUKERNELS_CUH
#define GPUKERNELS_CUH

#include <string>
#include <functional>

// GPU data structures (POD types for CUDA)
struct StationData_GPU {
    double latitude;
    double longitude;
};

struct VelocityLayer1D_GPU {
    double vp;
    double maxDepth;
};

struct GridSearchParams {
    double xMin, xMax;
    double yMin, yMax;
    double depthMin, depthMax;
    double gridSpacing;
    int nX, nY, nZ;
    
    double refLat, refLon;
    double homogeneousVp;
    
    int nStations;
    int nLayers;
};

struct GridSearchResult {
    double x, y, z;
    double misfit;
    bool converged;
};

struct GPUDeviceInfo {
    bool available;
    std::string deviceName;
    int computeCapability;
    size_t totalMemory;
    int maxThreadsPerBlock;
    int multiProcessorCount;
};

// Progress callback type
using ProgressCallback = std::function<void(int, const char*)>;

// Host functions
extern "C" {
    bool initializeGPU(GPUDeviceInfo &info);
    
    bool launchGridSearchGPU(const GridSearchParams &params,
                            const StationData_GPU *h_stations,
                            const double *h_observedTimes,
                            const VelocityLayer1D_GPU *h_layers1D,
                            GridSearchResult &result,
                            ProgressCallback callback = nullptr);
    
    bool launchMonteCarloSearchGPU(const GridSearchParams &params,
                                  const StationData_GPU *h_stations,
                                  const double *h_observedTimes,
                                  const VelocityLayer1D_GPU *h_layers1D,
                                  int nSamples,
                                  GridSearchResult &result,
                                  ProgressCallback callback = nullptr);
    
    bool launchGaussNewtonGPU(const GridSearchParams &params,
                             const StationData_GPU *h_stations,
                             const double *h_observedTimes,
                             const VelocityLayer1D_GPU *h_layers1D,
                             double tolerance,
                             int maxIter,
                             GridSearchResult &result,
                             ProgressCallback callback = nullptr);
}

#endif // GPUKERNELS_CUH
