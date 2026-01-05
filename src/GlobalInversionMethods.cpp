#include "InversionMethods.h"
#include <cmath>
#include <chrono>
#include <algorithm>
#include <random>
#include <iostream>
#include <limits>

// Constants
constexpr double EARTH_RADIUS = 6371.0; // km
constexpr double DEG_TO_RAD = M_PI / 180.0;

// =============================================================================
// Helper Functions
// =============================================================================

double GlobalInversionMethod::computeTravelTime(
    double srcX, double srcY, double srcZ,
    double staLat, double staLon,
    const VelocityModel* vm) const 
{
    double srcLat = srcY;
    double srcLon = srcX;
    double srcDepth = srcZ;
    
    // Haversine distance
    double dLat = (staLat - srcLat) * DEG_TO_RAD;
    double dLon = (staLon - srcLon) * DEG_TO_RAD;
    double lat1 = srcLat * DEG_TO_RAD;
    double lat2 = staLat * DEG_TO_RAD;
    
    double a = sin(dLat/2) * sin(dLat/2) +
               cos(lat1) * cos(lat2) * sin(dLon/2) * sin(dLon/2);
    double c = 2 * atan2(sqrt(a), sqrt(1-a));
    double epicentralDist = EARTH_RADIUS * c;
    
    double distance3D = sqrt(epicentralDist * epicentralDist + srcDepth * srcDepth);
    
    double avgLat = (srcLat + staLat) / 2.0;
    double avgLon = (srcLon + staLon) / 2.0;
    double avgDepth = srcDepth / 2.0;
    double velocity = vm->getVelocity(avgLat, avgLon, avgDepth);
    
    return distance3D / velocity;
}

double GlobalInversionMethod::computeMisfit(
    double x, double y, double z, double t0,
    const std::vector<Station>& stations,
    const VelocityModel* vm) const 
{
    double misfit = 0.0;
    for (const auto& sta : stations) {
        double travelTime = computeTravelTime(x, y, z, sta.lat, sta.lon, vm);
        double predictedArrival = t0 + travelTime;
        double residual = sta.arrivalTime - predictedArrival;
        misfit += residual * residual;
    }
    return sqrt(misfit / stations.size());
}

// =============================================================================
// GRID SEARCH METHOD
// =============================================================================

GridSearchMethod::GridSearchMethod(bool useMonteCarlo, int sampleSize)
    : monteCarloSampling(useMonteCarlo), numSamples(sampleSize)
{
    std::random_device rd;
    rng.seed(rd());
}

InversionResult GridSearchMethod::solve(
    const std::vector<Station>& stations,
    const SearchBoundary& boundary,
    const VelocityModel* velocityModel)
{
    auto startTime = std::chrono::high_resolution_clock::now();
    
    InversionResult result;
    result.converged = true;
    result.misfit = std::numeric_limits<double>::max();
    
    std::cout << "Starting Grid Search Method..." << std::endl;
    
    if (monteCarloSampling) {
        // Monte Carlo Random Sampling
        std::cout << "Using Monte Carlo sampling with " << numSamples << " samples" << std::endl;
        
        std::uniform_real_distribution<double> distX(boundary.xMin, boundary.xMax);
        std::uniform_real_distribution<double> distY(boundary.yMin, boundary.yMax);
        std::uniform_real_distribution<double> distZ(boundary.zMin, boundary.zMax);
        
        // Estimate origin time range from first arrival
        double minArrival = std::numeric_limits<double>::max();
        for (const auto& sta : stations) {
            minArrival = std::min(minArrival, sta.arrivalTime);
        }
        std::uniform_real_distribution<double> distT0(minArrival - 100, minArrival);
        
        for (int i = 0; i < numSamples; ++i) {
            double x = distX(rng);
            double y = distY(rng);
            double z = distZ(rng);
            double t0 = distT0(rng);
            
            double misfit = computeMisfit(x, y, z, t0, stations, velocityModel);
            
            if (misfit < result.misfit) {
                result.misfit = misfit;
                result.x = x;
                result.y = y;
                result.z = z;
                result.originTime = t0;
            }
            
            if ((i + 1) % 100 == 0) {
                std::cout << "Evaluated " << (i + 1) << "/" << numSamples 
                         << " samples, best misfit = " << result.misfit << std::endl;
            }
        }
        
        result.iterations = numSamples;
        
    } else {
        // Exhaustive Grid Search
        std::cout << "Using exhaustive grid search" << std::endl;
        
        // Calculate grid points
        int nX = std::max(1, static_cast<int>((boundary.xMax - boundary.xMin) / boundary.gridSpacing)) + 1;
        int nY = std::max(1, static_cast<int>((boundary.yMax - boundary.yMin) / boundary.gridSpacing)) + 1;
        int nZ = std::max(1, static_cast<int>((boundary.zMax - boundary.zMin) / boundary.gridSpacing)) + 1;
        
        std::cout << "Grid size: " << nX << " × " << nY << " × " << nZ 
                 << " = " << (nX * nY * nZ) << " points" << std::endl;
        
        // Estimate origin time from first arrival
        double minArrival = std::numeric_limits<double>::max();
        for (const auto& sta : stations) {
            minArrival = std::min(minArrival, sta.arrivalTime);
        }
        
        int totalPoints = nX * nY * nZ;
        int evaluated = 0;
        
        for (int ix = 0; ix < nX; ++ix) {
            double x = boundary.xMin + ix * boundary.gridSpacing;
            
            for (int iy = 0; iy < nY; ++iy) {
                double y = boundary.yMin + iy * boundary.gridSpacing;
                
                for (int iz = 0; iz < nZ; ++iz) {
                    double z = boundary.zMin + iz * boundary.gridSpacing;
                    
                    // Estimate origin time from nearest station
                    double t0 = minArrival - 50; // Initial guess
                    
                    double misfit = computeMisfit(x, y, z, t0, stations, velocityModel);
                    
                    if (misfit < result.misfit) {
                        result.misfit = misfit;
                        result.x = x;
                        result.y = y;
                        result.z = z;
                        result.originTime = t0;
                    }
                    
                    evaluated++;
                    if (evaluated % 1000 == 0) {
                        std::cout << "Evaluated " << evaluated << "/" << totalPoints 
                                 << " points (" << (100.0 * evaluated / totalPoints) 
                                 << "%), best misfit = " << result.misfit << std::endl;
                    }
                }
            }
        }
        
        result.iterations = totalPoints;
    }
    
    result.misfitHistory.push_back(result.misfit);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;
    result.computationTime = elapsed.count();
    
    std::cout << "Grid Search completed!" << std::endl;
    std::cout << "Best location: (" << result.x << ", " << result.y << ", " << result.z << ")" << std::endl;
    std::cout << "Best misfit: " << result.misfit << std::endl;
    
    return result;
}

// =============================================================================
// SIMULATED ANNEALING METHOD
// =============================================================================

SimulatedAnnealingMethod::SimulatedAnnealingMethod(
    SAVariant variant,
    double T0, double Tf,
    CoolingSchedule schedule,
    double alpha,
    int iterPerTemp,
    int maxIter)
    : variant(variant), initialTemp(T0), finalTemp(Tf),
      coolingSchedule(schedule), coolingAlpha(alpha),
      iterationsPerTemp(iterPerTemp), maxIterations(maxIter)
{
    std::random_device rd;
    rng.seed(rd());
}

double SimulatedAnnealingMethod::updateTemperature(double T, int iteration) const {
    switch (coolingSchedule) {
        case CoolingSchedule::Exponential:
            return initialTemp * pow(coolingAlpha, iteration / iterationsPerTemp);
            
        case CoolingSchedule::Linear:
            return initialTemp - coolingAlpha * iteration;
            
        case CoolingSchedule::Logarithmic:
            return initialTemp / (1.0 + coolingAlpha * log(1.0 + iteration));
            
        case CoolingSchedule::Inverse:
            return initialTemp / (1.0 + coolingAlpha * iteration);
            
        case CoolingSchedule::Adaptive:
            return initialTemp * pow(1.0 - (double)iteration / maxIterations, coolingAlpha);
            
        case CoolingSchedule::CauchySchedule:
            return initialTemp / (1.0 + iteration);
            
        case CoolingSchedule::FastCauchy:
            return initialTemp / (1.0 + coolingAlpha * iteration);
            
        case CoolingSchedule::VeryFast:
            return initialTemp * exp(-coolingAlpha * pow(iteration, 1.0/3.0));
            
        default:
            return T * coolingAlpha;
    }
}

bool SimulatedAnnealingMethod::acceptSolution(
    double currentMisfit, double newMisfit, double T) const 
{
    if (newMisfit < currentMisfit) {
        return true; // Always accept better solution
    }
    
    double delta = newMisfit - currentMisfit;
    double probability = exp(-delta / T);
    
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(const_cast<std::mt19937&>(rng)) < probability;
}

void SimulatedAnnealingMethod::generateNeighbor(
    double& x, double& y, double& z, double& t0,
    const SearchBoundary& boundary, double T) const 
{
    std::normal_distribution<double> dist(0.0, 1.0);
    
    // Step size proportional to temperature and search space
    double stepX = (boundary.xMax - boundary.xMin) * 0.1 * (T / initialTemp);
    double stepY = (boundary.yMax - boundary.yMin) * 0.1 * (T / initialTemp);
    double stepZ = (boundary.zMax - boundary.zMin) * 0.1 * (T / initialTemp);
    double stepT0 = 10.0 * (T / initialTemp);
    
    // Generate neighbor with Gaussian perturbation
    x += dist(const_cast<std::mt19937&>(rng)) * stepX;
    y += dist(const_cast<std::mt19937&>(rng)) * stepY;
    z += dist(const_cast<std::mt19937&>(rng)) * stepZ;
    t0 += dist(const_cast<std::mt19937&>(rng)) * stepT0;
    
    // Apply boundary constraints
    x = std::max(boundary.xMin, std::min(boundary.xMax, x));
    y = std::max(boundary.yMin, std::min(boundary.yMax, y));
    z = std::max(boundary.zMin, std::min(boundary.zMax, z));
}

InversionResult SimulatedAnnealingMethod::solve(
    const std::vector<Station>& stations,
    const SearchBoundary& boundary,
    const VelocityModel* velocityModel)
{
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Initialize random solution
    std::uniform_real_distribution<double> distX(boundary.xMin, boundary.xMax);
    std::uniform_real_distribution<double> distY(boundary.yMin, boundary.yMax);
    std::uniform_real_distribution<double> distZ(boundary.zMin, boundary.zMax);
    
    double minArrival = std::numeric_limits<double>::max();
    for (const auto& sta : stations) {
        minArrival = std::min(minArrival, sta.arrivalTime);
    }
    std::uniform_real_distribution<double> distT0(minArrival - 100, minArrival);
    
    InversionResult result;
    result.x = distX(rng);
    result.y = distY(rng);
    result.z = distZ(rng);
    result.originTime = distT0(rng);
    result.converged = false;
    
    double currentMisfit = computeMisfit(result.x, result.y, result.z, 
                                         result.originTime, stations, velocityModel);
    
    double bestMisfit = currentMisfit;
    double bestX = result.x;
    double bestY = result.y;
    double bestZ = result.z;
    double bestT0 = result.originTime;
    
    double T = initialTemp;
    int acceptedMoves = 0;
    int totalMoves = 0;
    
    std::cout << "Starting Simulated Annealing..." << std::endl;
    std::cout << "Initial temperature: " << T << std::endl;
    
    for (int iter = 0; iter < maxIterations; ++iter) {
        // Generate neighbor
        double newX = result.x;
        double newY = result.y;
        double newZ = result.z;
        double newT0 = result.originTime;
        
        generateNeighbor(newX, newY, newZ, newT0, boundary, T);
        
        double newMisfit = computeMisfit(newX, newY, newZ, newT0, stations, velocityModel);
        
        // Acceptance criterion
        if (acceptSolution(currentMisfit, newMisfit, T)) {
            result.x = newX;
            result.y = newY;
            result.z = newZ;
            result.originTime = newT0;
            currentMisfit = newMisfit;
            acceptedMoves++;
            
            // Update best solution
            if (newMisfit < bestMisfit) {
                bestMisfit = newMisfit;
                bestX = newX;
                bestY = newY;
                bestZ = newZ;
                bestT0 = newT0;
            }
        }
        
        totalMoves++;
        
        // Update temperature
        if (iter % iterationsPerTemp == 0) {
            T = updateTemperature(T, iter);
            
            double acceptanceRatio = (double)acceptedMoves / totalMoves;
            std::cout << "Iter " << iter << ": T = " << T 
                     << ", best misfit = " << bestMisfit
                     << ", acceptance = " << acceptanceRatio << std::endl;
            
            acceptedMoves = 0;
            totalMoves = 0;
            
            result.misfitHistory.push_back(bestMisfit);
        }
        
        // Check if temperature is too low
        if (T < finalTemp) {
            result.converged = true;
            std::cout << "Temperature reached final value" << std::endl;
            break;
        }
    }
    
    // Return best solution found
    result.x = bestX;
    result.y = bestY;
    result.z = bestZ;
    result.originTime = bestT0;
    result.misfit = bestMisfit;
    result.iterations = maxIterations;
    
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;
    result.computationTime = elapsed.count();
    
    std::cout << "Simulated Annealing completed!" << std::endl;
    std::cout << "Best location: (" << result.x << ", " << result.y << ", " << result.z << ")" << std::endl;
    std::cout << "Best misfit: " << result.misfit << std::endl;
    
    return result;
}

// =============================================================================
// VELOCITY MODEL IMPLEMENTATIONS
// =============================================================================

ThreeDVelocityModel::ThreeDVelocityModel(
    const std::vector<GridPoint>& grid,
    const SearchBoundary& boundary)
    : gridData(grid), bounds(boundary)
{}

double ThreeDVelocityModel::getVelocity(double lat, double lon, double depth) const {
    // Simple nearest neighbor for now
    // TODO: Implement trilinear interpolation
    
    double minDist = std::numeric_limits<double>::max();
    double nearestVp = 6.0; // Default
    
    for (const auto& point : gridData) {
        double dist = sqrt(
            pow(lat - point.lat, 2) + 
            pow(lon - point.lon, 2) + 
            pow(depth - point.depth, 2)
        );
        
        if (dist < minDist) {
            minDist = dist;
            nearestVp = point.vp;
        }
    }
    
    return nearestVp;
}

double ThreeDVelocityModel::interpolate(double lat, double lon, double depth) const {
    // TODO: Implement trilinear interpolation
    return getVelocity(lat, lon, depth);
}
