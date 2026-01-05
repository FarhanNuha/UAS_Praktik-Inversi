#include "CUDAInversionWrapper.h"
#include <chrono>
#include <iostream>
#include <limits>

#ifdef USE_CUDA

// Helper function to convert Station vector to float array
std::vector<float> stationsToFloatArray(const std::vector<Station>& stations) {
    std::vector<float> data;
    data.reserve(stations.size() * 3);
    for (const auto& sta : stations) {
        data.push_back(static_cast<float>(sta.lat));
        data.push_back(static_cast<float>(sta.lon));
        data.push_back(static_cast<float>(sta.arrivalTime));
    }
    return data;
}

// =============================================================================
// CUDA Grid Search
// =============================================================================

CUDAGridSearchMethod::CUDAGridSearchMethod(bool useMonteCarlo, int sampleSize)
    : monteCarloSampling(useMonteCarlo), numSamples(sampleSize)
{}

InversionResult CUDAGridSearchMethod::solve(
    const std::vector<Station>& stations,
    const SearchBoundary& boundary,
    const VelocityModel* velocityModel)
{
    auto startTime = std::chrono::high_resolution_clock::now();
    
    InversionResult result;
    result.converged = true;
    
    std::cout << "Starting CUDA Grid Search..." << std::endl;
    
    // Convert stations to float array
    auto stationData = stationsToFloatArray(stations);
    
    // Get homogeneous velocity (assuming homogeneous model for GPU)
    const HomogeneousVelocityModel* homogModel = 
        dynamic_cast<const HomogeneousVelocityModel*>(velocityModel);
    float vp = homogModel ? 6.0f : 6.0f; // Default to 6.0 km/s
    
    float bestParams[5];
    
    if (monteCarloSampling) {
        std::cout << "Using Monte Carlo with " << numSamples << " samples" << std::endl;
        
        cudaMonteCarloSearch(
            stationData.data(), stations.size(),
            boundary.xMin, boundary.xMax,
            boundary.yMin, boundary.yMax,
            boundary.zMin, boundary.zMax,
            vp, numSamples, bestParams
        );
        
        result.iterations = numSamples;
    } else {
        int nX = static_cast<int>((boundary.xMax - boundary.xMin) / boundary.gridSpacing) + 1;
        int nY = static_cast<int>((boundary.yMax - boundary.yMin) / boundary.gridSpacing) + 1;
        int nZ = static_cast<int>((boundary.zMax - boundary.zMin) / boundary.gridSpacing) + 1;
        
        std::cout << "Using exhaustive grid: " << nX << "×" << nY << "×" << nZ << std::endl;
        
        cudaGridSearch(
            stationData.data(), stations.size(),
            boundary.xMin, boundary.xMax, nX,
            boundary.yMin, boundary.yMax, nY,
            boundary.zMin, boundary.zMax, nZ,
            boundary.gridSpacing, vp,
            bestParams, nullptr
        );
        
        result.iterations = nX * nY * nZ;
    }
    
    result.x = bestParams[0];
    result.y = bestParams[1];
    result.z = bestParams[2];
    result.originTime = bestParams[3];
    result.misfit = bestParams[4];
    result.misfitHistory.push_back(result.misfit);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;
    result.computationTime = elapsed.count();
    
    std::cout << "CUDA Grid Search completed!" << std::endl;
    std::cout << "Best: (" << result.x << ", " << result.y << ", " << result.z << ")" << std::endl;
    std::cout << "Misfit: " << result.misfit << std::endl;
    
    return result;
}

// =============================================================================
// CUDA Simple SA
// =============================================================================

CUDASimpleSAMethod::CUDASimpleSAMethod(
    double T0, double Tf, double alpha, int iterPerTemp, int maxIter,
    int coolingSchedule, int numChains)
    : initialTemp(T0), finalTemp(Tf), coolingAlpha(alpha),
      iterationsPerTemp(iterPerTemp), maxIterations(maxIter),
      coolingScheduleType(coolingSchedule), numChains(numChains)
{}

InversionResult CUDASimpleSAMethod::solve(
    const std::vector<Station>& stations,
    const SearchBoundary& boundary,
    const VelocityModel* velocityModel)
{
    auto startTime = std::chrono::high_resolution_clock::now();
    
    InversionResult result;
    
    std::cout << "Starting CUDA Simple SA with " << numChains << " chains..." << std::endl;
    
    auto stationData = stationsToFloatArray(stations);
    const HomogeneousVelocityModel* homogModel = 
        dynamic_cast<const HomogeneousVelocityModel*>(velocityModel);
    float vp = homogModel ? 6.0f : 6.0f;
    
    float bestSolution[5];
    std::vector<float> misfitHistory(maxIterations);
    
    cudaSimpleSA(
        stationData.data(), stations.size(),
        boundary.xMin, boundary.xMax,
        boundary.yMin, boundary.yMax,
        boundary.zMin, boundary.zMax,
        vp,
        initialTemp, finalTemp, coolingAlpha,
        iterationsPerTemp, maxIterations, coolingScheduleType,
        numChains, bestSolution, misfitHistory.data()
    );
    
    result.x = bestSolution[0];
    result.y = bestSolution[1];
    result.z = bestSolution[2];
    result.originTime = bestSolution[3];
    result.misfit = bestSolution[4];
    result.iterations = maxIterations;
    result.converged = true;
    
    // Copy misfit history (sample every 100 iterations)
    for (int i = 0; i < maxIterations; i += 100) {
        result.misfitHistory.push_back(misfitHistory[i]);
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;
    result.computationTime = elapsed.count();
    
    std::cout << "CUDA Simple SA completed!" << std::endl;
    std::cout << "Best: (" << result.x << ", " << result.y << ", " << result.z << ")" << std::endl;
    std::cout << "Misfit: " << result.misfit << std::endl;
    
    return result;
}

// =============================================================================
// CUDA Metropolis SA
// =============================================================================

CUDAMetropolisSAMethod::CUDAMetropolisSAMethod(
    double T0, double Tf, double alpha, int markovChainLen, double targetAcc,
    int maxIter, int coolingSchedule, int numChains)
    : initialTemp(T0), finalTemp(Tf), coolingAlpha(alpha),
      markovChainLength(markovChainLen), targetAcceptance(targetAcc),
      maxIterations(maxIter), coolingScheduleType(coolingSchedule), numChains(numChains)
{}

InversionResult CUDAMetropolisSAMethod::solve(
    const std::vector<Station>& stations,
    const SearchBoundary& boundary,
    const VelocityModel* velocityModel)
{
    auto startTime = std::chrono::high_resolution_clock::now();
    
    InversionResult result;
    
    std::cout << "Starting CUDA Metropolis SA with " << numChains << " chains..." << std::endl;
    
    auto stationData = stationsToFloatArray(stations);
    const HomogeneousVelocityModel* homogModel = 
        dynamic_cast<const HomogeneousVelocityModel*>(velocityModel);
    float vp = homogModel ? 6.0f : 6.0f;
    
    float bestSolution[5];
    std::vector<float> misfitHistory(maxIterations);
    float acceptanceRate = 0.0f;
    
    cudaMetropolisSA(
        stationData.data(), stations.size(),
        boundary.xMin, boundary.xMax,
        boundary.yMin, boundary.yMax,
        boundary.zMin, boundary.zMax,
        vp,
        initialTemp, finalTemp, coolingAlpha,
        markovChainLength, targetAcceptance,
        maxIterations, coolingScheduleType, numChains,
        bestSolution, misfitHistory.data(), &acceptanceRate
    );
    
    result.x = bestSolution[0];
    result.y = bestSolution[1];
    result.z = bestSolution[2];
    result.originTime = bestSolution[3];
    result.misfit = bestSolution[4];
    result.iterations = maxIterations;
    result.converged = true;
    
    for (int i = 0; i < maxIterations; i += 100) {
        result.misfitHistory.push_back(misfitHistory[i]);
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;
    result.computationTime = elapsed.count();
    
    std::cout << "CUDA Metropolis SA completed!" << std::endl;
    std::cout << "Acceptance rate: " << (acceptanceRate * 100) << "%" << std::endl;
    std::cout << "Best: (" << result.x << ", " << result.y << ", " << result.z << ")" << std::endl;
    std::cout << "Misfit: " << result.misfit << std::endl;
    
    return result;
}

// =============================================================================
// CUDA Cauchy SA
// =============================================================================

CUDACauchySAMethod::CUDACauchySAMethod(
    double T0, double Tf, double alpha, int dim, double initStepSize,
    int iterPerTemp, int maxIter, int coolingSchedule, int numChains)
    : initialTemp(T0), finalTemp(Tf), coolingAlpha(alpha),
      dimension(dim), initialStepSize(initStepSize),
      iterationsPerTemp(iterPerTemp), maxIterations(maxIter),
      coolingScheduleType(coolingSchedule), numChains(numChains)
{}

InversionResult CUDACauchySAMethod::solve(
    const std::vector<Station>& stations,
    const SearchBoundary& boundary,
    const VelocityModel* velocityModel)
{
    auto startTime = std::chrono::high_resolution_clock::now();
    
    InversionResult result;
    
    std::cout << "Starting CUDA Cauchy SA with " << numChains << " chains..." << std::endl;
    
    auto stationData = stationsToFloatArray(stations);
    const HomogeneousVelocityModel* homogModel = 
        dynamic_cast<const HomogeneousVelocityModel*>(velocityModel);
    float vp = homogModel ? 6.0f : 6.0f;
    
    float bestSolution[5];
    std::vector<float> misfitHistory(maxIterations);
    
    cudaCauchySA(
        stationData.data(), stations.size(),
        boundary.xMin, boundary.xMax,
        boundary.yMin, boundary.yMax,
        boundary.zMin, boundary.zMax,
        vp,
        initialTemp, finalTemp, coolingAlpha,
        dimension, initialStepSize,
        iterationsPerTemp, maxIterations, coolingScheduleType,
        numChains, bestSolution, misfitHistory.data()
    );
    
    result.x = bestSolution[0];
    result.y = bestSolution[1];
    result.z = bestSolution[2];
    result.originTime = bestSolution[3];
    result.misfit = bestSolution[4];
    result.iterations = maxIterations;
    result.converged = true;
    
    for (int i = 0; i < maxIterations; i += 100) {
        result.misfitHistory.push_back(misfitHistory[i]);
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;
    result.computationTime = elapsed.count();
    
    std::cout << "CUDA Cauchy SA completed!" << std::endl;
    std::cout << "Best: (" << result.x << ", " << result.y << ", " << result.z << ")" << std::endl;
    std::cout << "Misfit: " << result.misfit << std::endl;
    
    return result;
}

// =============================================================================
// CUDA Standard GA
// =============================================================================

CUDAStandardGAMethod::CUDAStandardGAMethod(
    int popSize, int maxGen, double crossRate, double mutRate,
    int elite, int tournament)
    : populationSize(popSize), maxGenerations(maxGen),
      crossoverRate(crossRate), mutationRate(mutRate),
      eliteSize(elite), tournamentSize(tournament)
{}

InversionResult CUDAStandardGAMethod::solve(
    const std::vector<Station>& stations,
    const SearchBoundary& boundary,
    const VelocityModel* velocityModel)
{
    auto startTime = std::chrono::high_resolution_clock::now();
    
    InversionResult result;
    
    std::cout << "Starting CUDA Standard GA..." << std::endl;
    
    auto stationData = stationsToFloatArray(stations);
    const HomogeneousVelocityModel* homogModel = 
        dynamic_cast<const HomogeneousVelocityModel*>(velocityModel);
    float vp = homogModel ? 6.0f : 6.0f;
    
    float bestSolution[5];
    std::vector<float> generationBest(maxGenerations);
    
    cudaStandardGA(
        stationData.data(), stations.size(),
        boundary.xMin, boundary.xMax,
        boundary.yMin, boundary.yMax,
        boundary.zMin, boundary.zMax,
        vp,
        populationSize, maxGenerations,
        crossoverRate, mutationRate, eliteSize, tournamentSize,
        bestSolution, generationBest.data()
    );
    
    result.x = bestSolution[0];
    result.y = bestSolution[1];
    result.z = bestSolution[2];
    result.originTime = bestSolution[3];
    result.misfit = bestSolution[4];
    result.iterations = maxGenerations;
    result.converged = true;
    result.misfitHistory.assign(generationBest.begin(), generationBest.end());
    
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;
    result.computationTime = elapsed.count();
    
    std::cout << "CUDA Standard GA completed!" << std::endl;
    std::cout << "Best: (" << result.x << ", " << result.y << ", " << result.z << ")" << std::endl;
    std::cout << "Misfit: " << result.misfit << std::endl;
    
    return result;
}

// Similar implementations for other GA variants...
// (Steady State GA and SPEA2 follow same pattern)

CUDASteadyStateGAMethod::CUDASteadyStateGAMethod(
    int popSize, int maxIter, int offspring, double crossRate, double mutRate, int tournament)
    : populationSize(popSize), maxIterations(maxIter), offspringPerIteration(offspring),
      crossoverRate(crossRate), mutationRate(mutRate), tournamentSize(tournament)
{}

InversionResult CUDASteadyStateGAMethod::solve(
    const std::vector<Station>& stations,
    const SearchBoundary& boundary,
    const VelocityModel* velocityModel)
{
    auto startTime = std::chrono::high_resolution_clock::now();
    
    InversionResult result;
    
    std::cout << "Starting CUDA Steady State GA..." << std::endl;
    
    auto stationData = stationsToFloatArray(stations);
    const HomogeneousVelocityModel* homogModel = 
        dynamic_cast<const HomogeneousVelocityModel*>(velocityModel);
    float vp = homogModel ? 6.0f : 6.0f;
    
    float bestSolution[5];
    std::vector<float> iterationBest(maxIterations / 100);
    
    cudaSteadyStateGA(
        stationData.data(), stations.size(),
        boundary.xMin, boundary.xMax,
        boundary.yMin, boundary.yMax,
        boundary.zMin, boundary.zMax,
        vp,
        populationSize, maxIterations, offspringPerIteration,
        crossoverRate, mutationRate, tournamentSize,
        bestSolution, iterationBest.data()
    );
    
    result.x = bestSolution[0];
    result.y = bestSolution[1];
    result.z = bestSolution[2];
    result.originTime = bestSolution[3];
    result.misfit = bestSolution[4];
    result.iterations = maxIterations;
    result.converged = true;
    result.misfitHistory.assign(iterationBest.begin(), iterationBest.end());
    
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;
    result.computationTime = elapsed.count();
    
    std::cout << "CUDA Steady State GA completed!" << std::endl;
    
    return result;
}

CUDASPEA2Method::CUDASPEA2Method(
    int popSize, int archSize, int maxGen, double crossRate, double mutRate, int k)
    : populationSize(popSize), archiveSize(archSize), maxGenerations(maxGen),
      crossoverRate(crossRate), mutationRate(mutRate), kNearest(k)
{}

InversionResult CUDASPEA2Method::solve(
    const std::vector<Station>& stations,
    const SearchBoundary& boundary,
    const VelocityModel* velocityModel)
{
    auto startTime = std::chrono::high_resolution_clock::now();
    
    InversionResult result;
    
    std::cout << "Starting CUDA SPEA2..." << std::endl;
    
    auto stationData = stationsToFloatArray(stations);
    const HomogeneousVelocityModel* homogModel = 
        dynamic_cast<const HomogeneousVelocityModel*>(velocityModel);
    float vp = homogModel ? 6.0f : 6.0f;
    
    float bestSolution[5];
    std::vector<float> generationBest(maxGenerations);
    
    cudaSPEA2(
        stationData.data(), stations.size(),
        boundary.xMin, boundary.xMax,
        boundary.yMin, boundary.yMax,
        boundary.zMin, boundary.zMax,
        vp,
        populationSize, archiveSize, maxGenerations,
        crossoverRate, mutationRate, kNearest,
        bestSolution, generationBest.data()
    );
    
    result.x = bestSolution[0];
    result.y = bestSolution[1];
    result.z = bestSolution[2];
    result.originTime = bestSolution[3];
    result.misfit = bestSolution[4];
    result.iterations = maxGenerations;
    result.converged = true;
    result.misfitHistory.assign(generationBest.begin(), generationBest.end());
    
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;
    result.computationTime = elapsed.count();
    
    std::cout << "CUDA SPEA2 completed!" << std::endl;
    
    return result;
}

#endif // USE_CUDA
