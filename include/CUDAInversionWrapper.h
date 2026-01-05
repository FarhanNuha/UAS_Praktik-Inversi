#ifndef CUDAINVERSIONWRAPPER_H
#define CUDAINVERSIONWRAPPER_H

#include "InversionMethods.h"
#include <vector>

#ifdef USE_CUDA

// Forward declarations of CUDA functions
extern "C" {
    // Grid Search
    void cudaGridSearch(
        const float* h_stations, int numStations,
        float xMin, float xMax, int nX,
        float yMin, float yMax, int nY,
        float zMin, float zMax, int nZ,
        float gridSpacing, float homogeneousVp,
        float* h_bestParams, float* h_misfitGrid);
    
    void cudaMonteCarloSearch(
        const float* h_stations, int numStations,
        float xMin, float xMax, float yMin, float yMax, float zMin, float zMax,
        float homogeneousVp, int numSamples, float* h_bestParams);
    
    // Simulated Annealing
    void cudaSimpleSA(
        const float* h_stations, int numStations,
        float xMin, float xMax, float yMin, float yMax, float zMin, float zMax,
        float homogeneousVp,
        float T0, float Tf, float alpha, int iterPerTemp, int maxIter, int coolingSchedule,
        int numChains, float* h_bestSolution, float* h_misfitHistory);
    
    void cudaMetropolisSA(
        const float* h_stations, int numStations,
        float xMin, float xMax, float yMin, float yMax, float zMin, float zMax,
        float homogeneousVp,
        float T0, float Tf, float alpha, int markovChainLength, float targetAcceptance,
        int maxIter, int coolingSchedule, int numChains,
        float* h_bestSolution, float* h_misfitHistory, float* h_acceptanceRate);
    
    void cudaCauchySA(
        const float* h_stations, int numStations,
        float xMin, float xMax, float yMin, float yMax, float zMin, float zMax,
        float homogeneousVp,
        float T0, float Tf, float alpha, int dimension, float initialStepSize,
        int iterPerTemp, int maxIter, int coolingSchedule,
        int numChains, float* h_bestSolution, float* h_misfitHistory);
    
    // Genetic Algorithms
    void cudaStandardGA(
        const float* h_stations, int numStations,
        float xMin, float xMax, float yMin, float yMax, float zMin, float zMax,
        float homogeneousVp,
        int populationSize, int maxGenerations,
        float crossoverRate, float mutationRate, int eliteSize, int tournamentSize,
        float* h_bestSolution, float* h_generationBest);
    
    void cudaSteadyStateGA(
        const float* h_stations, int numStations,
        float xMin, float xMax, float yMin, float yMax, float zMin, float zMax,
        float homogeneousVp,
        int populationSize, int maxIterations, int offspringPerIter,
        float crossoverRate, float mutationRate, int tournamentSize,
        float* h_bestSolution, float* h_iterationBest);
    
    void cudaSPEA2(
        const float* h_stations, int numStations,
        float xMin, float xMax, float yMin, float yMax, float zMin, float zMax,
        float homogeneousVp,
        int populationSize, int archiveSize, int maxGenerations,
        float crossoverRate, float mutationRate, int kNearest,
        float* h_bestSolution, float* h_generationBest);
}

// =============================================================================
// C++ Wrapper Classes
// =============================================================================

class CUDAGridSearchMethod : public GlobalInversionMethod {
public:
    CUDAGridSearchMethod(bool useMonteCarlo = false, int sampleSize = 1000);
    
    InversionResult solve(
        const std::vector<Station>& stations,
        const SearchBoundary& boundary,
        const VelocityModel* velocityModel
    ) override;
    
private:
    bool monteCarloSampling;
    int numSamples;
};

class CUDASimpleSAMethod : public GlobalInversionMethod {
public:
    CUDASimpleSAMethod(
        double T0 = 1000.0, double Tf = 0.1, double alpha = 0.95,
        int iterPerTemp = 100, int maxIter = 10000,
        int coolingSchedule = 0, int numChains = 8
    );
    
    InversionResult solve(
        const std::vector<Station>& stations,
        const SearchBoundary& boundary,
        const VelocityModel* velocityModel
    ) override;
    
private:
    double initialTemp, finalTemp, coolingAlpha;
    int iterationsPerTemp, maxIterations, coolingScheduleType, numChains;
};

class CUDAMetropolisSAMethod : public GlobalInversionMethod {
public:
    CUDAMetropolisSAMethod(
        double T0 = 1500.0, double Tf = 0.05, double alpha = 0.90,
        int markovChainLength = 150, double targetAcceptance = 0.6,
        int maxIter = 15000, int coolingSchedule = 0, int numChains = 8
    );
    
    InversionResult solve(
        const std::vector<Station>& stations,
        const SearchBoundary& boundary,
        const VelocityModel* velocityModel
    ) override;
    
private:
    double initialTemp, finalTemp, coolingAlpha;
    int markovChainLength;
    double targetAcceptance;
    int maxIterations, coolingScheduleType, numChains;
};

class CUDACauchySAMethod : public GlobalInversionMethod {
public:
    CUDACauchySAMethod(
        double T0 = 2000.0, double Tf = 0.01, double alpha = 1.0,
        int dimension = 3, double initialStepSize = 1.0,
        int iterPerTemp = 200, int maxIter = 20000,
        int coolingSchedule = 0, int numChains = 8
    );
    
    InversionResult solve(
        const std::vector<Station>& stations,
        const SearchBoundary& boundary,
        const VelocityModel* velocityModel
    ) override;
    
private:
    double initialTemp, finalTemp, coolingAlpha;
    int dimension;
    double initialStepSize;
    int iterationsPerTemp, maxIterations, coolingScheduleType, numChains;
};

class CUDAStandardGAMethod : public GlobalInversionMethod {
public:
    CUDAStandardGAMethod(
        int popSize = 100, int maxGen = 200,
        double crossoverRate = 0.8, double mutationRate = 0.1,
        int eliteSize = 10, int tournamentSize = 5
    );
    
    InversionResult solve(
        const std::vector<Station>& stations,
        const SearchBoundary& boundary,
        const VelocityModel* velocityModel
    ) override;
    
private:
    int populationSize, maxGenerations;
    double crossoverRate, mutationRate;
    int eliteSize, tournamentSize;
};

class CUDASteadyStateGAMethod : public GlobalInversionMethod {
public:
    CUDASteadyStateGAMethod(
        int popSize = 100, int maxIter = 10000, int offspringPerIter = 2,
        double crossoverRate = 0.9, double mutationRate = 0.05,
        int tournamentSize = 3
    );
    
    InversionResult solve(
        const std::vector<Station>& stations,
        const SearchBoundary& boundary,
        const VelocityModel* velocityModel
    ) override;
    
private:
    int populationSize, maxIterations, offspringPerIteration;
    double crossoverRate, mutationRate;
    int tournamentSize;
};

class CUDASPEA2Method : public GlobalInversionMethod {
public:
    CUDASPEA2Method(
        int popSize = 100, int archiveSize = 100, int maxGen = 250,
        double crossoverRate = 0.9, double mutationRate = 0.1,
        int kNearest = 5
    );
    
    InversionResult solve(
        const std::vector<Station>& stations,
        const SearchBoundary& boundary,
        const VelocityModel* velocityModel
    ) override;
    
private:
    int populationSize, archiveSize, maxGenerations;
    double crossoverRate, mutationRate;
    int kNearest;
};

#endif // USE_CUDA

#endif // CUDAINVERSIONWRAPPER_H
