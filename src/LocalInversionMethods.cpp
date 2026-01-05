#include "InversionMethods.h"
#include <cmath>
#include <chrono>
#include <algorithm>
#include <iostream>

// Constants
constexpr double EARTH_RADIUS = 6371.0; // km
constexpr double DEG_TO_RAD = M_PI / 180.0;

// =============================================================================
// Helper Functions
// =============================================================================

double LocalInversionMethod::computeTravelTime(
    double srcX, double srcY, double srcZ,
    double staLat, double staLon,
    const VelocityModel* vm) const 
{
    // Convert source coordinates to lat/lon (assuming srcX=lon, srcY=lat)
    double srcLat = srcY;
    double srcLon = srcX;
    double srcDepth = srcZ;
    
    // Haversine distance for epicentral distance
    double dLat = (staLat - srcLat) * DEG_TO_RAD;
    double dLon = (staLon - srcLon) * DEG_TO_RAD;
    double lat1 = srcLat * DEG_TO_RAD;
    double lat2 = staLat * DEG_TO_RAD;
    
    double a = sin(dLat/2) * sin(dLat/2) +
               cos(lat1) * cos(lat2) * sin(dLon/2) * sin(dLon/2);
    double c = 2 * atan2(sqrt(a), sqrt(1-a));
    double epicentralDist = EARTH_RADIUS * c;
    
    // 3D distance (Pythagoras with depth)
    double distance3D = sqrt(epicentralDist * epicentralDist + srcDepth * srcDepth);
    
    // Get average velocity along ray path
    double avgLat = (srcLat + staLat) / 2.0;
    double avgLon = (srcLon + staLon) / 2.0;
    double avgDepth = srcDepth / 2.0;
    double velocity = vm->getVelocity(avgLat, avgLon, avgDepth);
    
    // Travel time = distance / velocity
    return distance3D / velocity;
}

double LocalInversionMethod::computeMisfit(
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
    return sqrt(misfit / stations.size()); // RMS
}

void LocalInversionMethod::computeJacobian(
    double x, double y, double z, double t0,
    const std::vector<Station>& stations,
    const VelocityModel* vm,
    std::vector<std::vector<double>>& J) const 
{
    const double h = 1e-4; // Small perturbation
    int n = stations.size();
    
    J.resize(n, std::vector<double>(4, 0.0));
    
    for (int i = 0; i < n; ++i) {
        const auto& sta = stations[i];
        double t_base = t0 + computeTravelTime(x, y, z, sta.lat, sta.lon, vm);
        
        // Partial derivative w.r.t. x
        double t_px = t0 + computeTravelTime(x+h, y, z, sta.lat, sta.lon, vm);
        J[i][0] = (t_px - t_base) / h;
        
        // Partial derivative w.r.t. y
        double t_py = t0 + computeTravelTime(x, y+h, z, sta.lat, sta.lon, vm);
        J[i][1] = (t_py - t_base) / h;
        
        // Partial derivative w.r.t. z
        double t_pz = t0 + computeTravelTime(x, y, z+h, sta.lat, sta.lon, vm);
        J[i][2] = (t_pz - t_base) / h;
        
        // Partial derivative w.r.t. t0
        J[i][3] = 1.0;
    }
}

// Matrix operations helpers
std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& A) {
    int m = A.size();
    int n = A[0].size();
    std::vector<std::vector<double>> AT(n, std::vector<double>(m));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            AT[j][i] = A[i][j];
    return AT;
}

std::vector<std::vector<double>> matmul(
    const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B) 
{
    int m = A.size();
    int n = B[0].size();
    int k = A[0].size();
    std::vector<std::vector<double>> C(m, std::vector<double>(n, 0.0));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            for (int p = 0; p < k; ++p)
                C[i][j] += A[i][p] * B[p][j];
    return C;
}

std::vector<double> matvec(
    const std::vector<std::vector<double>>& A,
    const std::vector<double>& v) 
{
    int m = A.size();
    int n = A[0].size();
    std::vector<double> result(m, 0.0);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            result[i] += A[i][j] * v[j];
    return result;
}

// Simple matrix inversion (Gauss-Jordan for small matrices)
std::vector<std::vector<double>> inverse(const std::vector<std::vector<double>>& A) {
    int n = A.size();
    std::vector<std::vector<double>> augmented(n, std::vector<double>(2*n, 0.0));
    
    // Create augmented matrix [A | I]
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            augmented[i][j] = A[i][j];
        }
        augmented[i][n + i] = 1.0;
    }
    
    // Gauss-Jordan elimination
    for (int i = 0; i < n; ++i) {
        // Find pivot
        double pivot = augmented[i][i];
        if (fabs(pivot) < 1e-10) {
            // Matrix is singular or nearly singular
            return std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
        }
        
        // Scale row
        for (int j = 0; j < 2*n; ++j) {
            augmented[i][j] /= pivot;
        }
        
        // Eliminate column
        for (int k = 0; k < n; ++k) {
            if (k != i) {
                double factor = augmented[k][i];
                for (int j = 0; j < 2*n; ++j) {
                    augmented[k][j] -= factor * augmented[i][j];
                }
            }
        }
    }
    
    // Extract inverse matrix
    std::vector<std::vector<double>> inv(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            inv[i][j] = augmented[i][n + j];
        }
    }
    
    return inv;
}

// =============================================================================
// GAUSS-NEWTON METHOD
// =============================================================================

GaussNewtonMethod::GaussNewtonMethod(int maxIter, double tol, double damping)
    : maxIterations(maxIter), tolerance(tol), dampingFactor(damping)
{}

InversionResult GaussNewtonMethod::solve(
    const std::vector<Station>& stations,
    const SearchBoundary& boundary,
    const VelocityModel* velocityModel,
    double initialX, double initialY, double initialZ,
    double initialOriginTime)
{
    auto startTime = std::chrono::high_resolution_clock::now();
    
    InversionResult result;
    result.x = initialX;
    result.y = initialY;
    result.z = initialZ;
    result.originTime = initialOriginTime;
    result.converged = false;
    
    std::cout << "Starting Gauss-Newton Method..." << std::endl;
    
    for (int iter = 0; iter < maxIterations; ++iter) {
        // Compute residuals
        std::vector<double> residuals(stations.size());
        for (size_t i = 0; i < stations.size(); ++i) {
            double travelTime = computeTravelTime(
                result.x, result.y, result.z,
                stations[i].lat, stations[i].lon, velocityModel
            );
            double predicted = result.originTime + travelTime;
            residuals[i] = stations[i].arrivalTime - predicted;
        }
        
        // Compute misfit
        double misfit = 0.0;
        for (double r : residuals) misfit += r * r;
        misfit = sqrt(misfit / stations.size());
        result.misfitHistory.push_back(misfit);
        
        std::cout << "Iter " << iter << ": misfit = " << misfit << std::endl;
        
        // Check convergence
        if (misfit < tolerance) {
            result.converged = true;
            std::cout << "Converged!" << std::endl;
            break;
        }
        
        // Compute Jacobian
        std::vector<std::vector<double>> J;
        computeJacobian(result.x, result.y, result.z, result.originTime,
                       stations, velocityModel, J);
        
        // Compute J^T * J
        auto JT = transpose(J);
        auto JTJ = matmul(JT, J);
        
        // Add damping (Tikhonov regularization)
        for (size_t i = 0; i < JTJ.size(); ++i) {
            JTJ[i][i] += dampingFactor;
        }
        
        // Compute J^T * residuals
        auto JTr = matvec(JT, residuals);
        
        // Solve (J^T * J) * delta = J^T * residuals
        auto JTJ_inv = inverse(JTJ);
        auto delta = matvec(JTJ_inv, JTr);
        
        // Update parameters
        result.x += delta[0];
        result.y += delta[1];
        result.z += delta[2];
        result.originTime += delta[3];
        
        // Apply boundary constraints
        result.x = std::max(boundary.xMin, std::min(boundary.xMax, result.x));
        result.y = std::max(boundary.yMin, std::min(boundary.yMax, result.y));
        result.z = std::max(boundary.zMin, std::min(boundary.zMax, result.z));
    }
    
    result.iterations = result.misfitHistory.size();
    result.misfit = result.misfitHistory.back();
    
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;
    result.computationTime = elapsed.count();
    
    return result;
}

// =============================================================================
// STEEPEST DESCENT METHOD
// =============================================================================

SteepestDescentMethod::SteepestDescentMethod(
    int maxIter, double tol, double stepSize, int lineSearchIter)
    : maxIterations(maxIter), tolerance(tol), 
      initialStepSize(stepSize), lineSearchIterations(lineSearchIter)
{}

InversionResult SteepestDescentMethod::solve(
    const std::vector<Station>& stations,
    const SearchBoundary& boundary,
    const VelocityModel* velocityModel,
    double initialX, double initialY, double initialZ,
    double initialOriginTime)
{
    auto startTime = std::chrono::high_resolution_clock::now();
    
    InversionResult result;
    result.x = initialX;
    result.y = initialY;
    result.z = initialZ;
    result.originTime = initialOriginTime;
    result.converged = false;
    
    std::cout << "Starting Steepest Descent Method..." << std::endl;
    
    for (int iter = 0; iter < maxIterations; ++iter) {
        // Compute current misfit
        double currentMisfit = computeMisfit(
            result.x, result.y, result.z, result.originTime,
            stations, velocityModel
        );
        result.misfitHistory.push_back(currentMisfit);
        
        std::cout << "Iter " << iter << ": misfit = " << currentMisfit << std::endl;
        
        // Check convergence
        if (currentMisfit < tolerance) {
            result.converged = true;
            std::cout << "Converged!" << std::endl;
            break;
        }
        
        // Compute gradient (numerical)
        const double h = 1e-4;
        double grad_x = (computeMisfit(result.x+h, result.y, result.z, result.originTime, stations, velocityModel)
                       - computeMisfit(result.x-h, result.y, result.z, result.originTime, stations, velocityModel)) / (2*h);
        double grad_y = (computeMisfit(result.x, result.y+h, result.z, result.originTime, stations, velocityModel)
                       - computeMisfit(result.x, result.y-h, result.z, result.originTime, stations, velocityModel)) / (2*h);
        double grad_z = (computeMisfit(result.x, result.y, result.z+h, result.originTime, stations, velocityModel)
                       - computeMisfit(result.x, result.y, result.z-h, result.originTime, stations, velocityModel)) / (2*h);
        double grad_t0 = (computeMisfit(result.x, result.y, result.z, result.originTime+h, stations, velocityModel)
                        - computeMisfit(result.x, result.y, result.z, result.originTime-h, stations, velocityModel)) / (2*h);
        
        // Line search for optimal step size
        double alpha = initialStepSize;
        for (int ls = 0; ls < lineSearchIterations; ++ls) {
            double newX = result.x - alpha * grad_x;
            double newY = result.y - alpha * grad_y;
            double newZ = result.z - alpha * grad_z;
            double newT0 = result.originTime - alpha * grad_t0;
            
            // Apply boundary constraints
            newX = std::max(boundary.xMin, std::min(boundary.xMax, newX));
            newY = std::max(boundary.yMin, std::min(boundary.yMax, newY));
            newZ = std::max(boundary.zMin, std::min(boundary.zMax, newZ));
            
            double newMisfit = computeMisfit(newX, newY, newZ, newT0, stations, velocityModel);
            
            if (newMisfit < currentMisfit) {
                result.x = newX;
                result.y = newY;
                result.z = newZ;
                result.originTime = newT0;
                break;
            }
            
            alpha *= 0.5; // Reduce step size
        }
    }
    
    result.iterations = result.misfitHistory.size();
    result.misfit = result.misfitHistory.back();
    
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;
    result.computationTime = elapsed.count();
    
    return result;
}

// =============================================================================
// LEVENBERG-MARQUARDT METHOD
// =============================================================================

LevenbergMarquardtMethod::LevenbergMarquardtMethod(
    int maxIter, double tol, double lambda, double lambdaUp, double lambdaDown)
    : maxIterations(maxIter), tolerance(tol), lambda(lambda),
      lambdaUpFactor(lambdaUp), lambdaDownFactor(lambdaDown)
{}

InversionResult LevenbergMarquardtMethod::solve(
    const std::vector<Station>& stations,
    const SearchBoundary& boundary,
    const VelocityModel* velocityModel,
    double initialX, double initialY, double initialZ,
    double initialOriginTime)
{
    auto startTime = std::chrono::high_resolution_clock::now();
    
    InversionResult result;
    result.x = initialX;
    result.y = initialY;
    result.z = initialZ;
    result.originTime = initialOriginTime;
    result.converged = false;
    
    double currentLambda = lambda;
    
    std::cout << "Starting Levenberg-Marquardt Method..." << std::endl;
    
    for (int iter = 0; iter < maxIterations; ++iter) {
        // Compute residuals
        std::vector<double> residuals(stations.size());
        for (size_t i = 0; i < stations.size(); ++i) {
            double travelTime = computeTravelTime(
                result.x, result.y, result.z,
                stations[i].lat, stations[i].lon, velocityModel
            );
            double predicted = result.originTime + travelTime;
            residuals[i] = stations[i].arrivalTime - predicted;
        }
        
        // Compute misfit
        double misfit = 0.0;
        for (double r : residuals) misfit += r * r;
        misfit = sqrt(misfit / stations.size());
        result.misfitHistory.push_back(misfit);
        
        std::cout << "Iter " << iter << ": misfit = " << misfit 
                  << ", λ = " << currentLambda << std::endl;
        
        // Check convergence
        if (misfit < tolerance) {
            result.converged = true;
            std::cout << "Converged!" << std::endl;
            break;
        }
        
        // Compute Jacobian
        std::vector<std::vector<double>> J;
        computeJacobian(result.x, result.y, result.z, result.originTime,
                       stations, velocityModel, J);
        
        // Compute J^T * J
        auto JT = transpose(J);
        auto JTJ = matmul(JT, J);
        
        // Add lambda to diagonal (Levenberg-Marquardt damping)
        for (size_t i = 0; i < JTJ.size(); ++i) {
            JTJ[i][i] += currentLambda;
        }
        
        // Compute J^T * residuals
        auto JTr = matvec(JT, residuals);
        
        // Solve (J^T * J + λI) * delta = J^T * residuals
        auto JTJ_inv = inverse(JTJ);
        auto delta = matvec(JTJ_inv, JTr);
        
        // Try update
        double newX = result.x + delta[0];
        double newY = result.y + delta[1];
        double newZ = result.z + delta[2];
        double newT0 = result.originTime + delta[3];
        
        // Apply boundary constraints
        newX = std::max(boundary.xMin, std::min(boundary.xMax, newX));
        newY = std::max(boundary.yMin, std::min(boundary.yMax, newY));
        newZ = std::max(boundary.zMin, std::min(boundary.zMax, newZ));
        
        double newMisfit = computeMisfit(newX, newY, newZ, newT0, stations, velocityModel);
        
        if (newMisfit < misfit) {
            // Accept update and decrease lambda
            result.x = newX;
            result.y = newY;
            result.z = newZ;
            result.originTime = newT0;
            currentLambda *= lambdaDownFactor;
        } else {
            // Reject update and increase lambda
            currentLambda *= lambdaUpFactor;
        }
    }
    
    result.iterations = result.misfitHistory.size();
    result.misfit = result.misfitHistory.back();
    
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;
    result.computationTime = elapsed.count();
    
    return result;
}
