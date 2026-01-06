#include "ComputationEngine.h"
#include <QDebug>
#include <algorithm>
#include <numeric>

#ifdef CUDA_ENABLED
#include "GPUKernels.cuh"
#endif

ComputationEngine::ComputationEngine(QObject *parent)
    : QObject(parent), boundarySet(false), stationsSet(false),
      velocityModelType("Homogen"), homogeneousVp(6.0),
      refLat(0.0), refLon(0.0), gpuInitialized(false), gpuAvailable(false)
{
    initializeGPU();
}

ComputationEngine::~ComputationEngine() {
    cleanupGPU();
}

void ComputationEngine::initializeGPU() {
#ifdef CUDA_ENABLED
    GPUDeviceInfo info;
    if (::initializeGPU(info)) {
        gpuAvailable = true;
        gpuInitialized = true;
        gpuDeviceName = QString::fromStdString(info.deviceName);
        qDebug() << "GPU initialized:" << gpuDeviceName;
        qDebug() << "Compute Capability:" << info.computeCapability;
        qDebug() << "Total Memory:" << info.totalMemory / (1024*1024) << "MB";
    } else {
        gpuAvailable = false;
        qDebug() << "GPU not available, using CPU mode";
    }
#else
    gpuAvailable = false;
    qDebug() << "CUDA not enabled, using CPU-only mode";
#endif
}

void ComputationEngine::cleanupGPU() {
#ifdef CUDA_ENABLED
    if (gpuInitialized) {
        gpuInitialized = false;
    }
#endif
}

bool ComputationEngine::isGPUAvailable() const {
    return gpuAvailable;
}

QString ComputationEngine::getGPUInfo() const {
    if (gpuAvailable) {
        return gpuDeviceName;
    }
    return "No GPU available";
}

void ComputationEngine::setBoundary(const BoundaryData &b) {
    boundary = b;
    boundarySet = true;
    refLat = (boundary.yMin + boundary.yMax) / 2.0;
    refLon = (boundary.xMin + boundary.xMax) / 2.0;
}

void ComputationEngine::setStations(const QVector<StationData> &s) {
    stations = s;
    stationsSet = true;
}

void ComputationEngine::setVelocityModel(const QString &type, double vp) {
    velocityModelType = type;
    homogeneousVp = vp;
}

void ComputationEngine::setVelocityModel1D(const QVector<VelocityLayer1D> &layers) {
    velocityModel1D = layers;
    velocityModelType = "1D";
}

void ComputationEngine::setVelocityModel3D(const QVector<VelocityPoint3D> &points) {
    velocityModel3D = points;
    velocityModelType = "3D";
}

void ComputationEngine::latLonToKm(double lat, double lon, double &x, double &y) const {
    double latRad = refLat * M_PI / 180.0;
    x = (lon - refLon) * 111.320 * cos(latRad);
    y = (lat - refLat) * 110.574;
}

void ComputationEngine::kmToLatLon(double x, double y, double &lat, double &lon) const {
    double latRad = refLat * M_PI / 180.0;
    lon = refLon + x / (111.320 * cos(latRad));
    lat = refLat + y / 110.574;
}

double ComputationEngine::haversineDistance(double lat1, double lon1, double lat2, double lon2) const {
    const double R = 6371.0;
    double dLat = (lat2 - lat1) * M_PI / 180.0;
    double dLon = (lon2 - lon1) * M_PI / 180.0;
    double a = sin(dLat/2) * sin(dLat/2) + cos(lat1 * M_PI / 180.0) * cos(lat2 * M_PI / 180.0) * sin(dLon/2) * sin(dLon/2);
    double c = 2 * atan2(sqrt(a), sqrt(1-a));
    return R * c;
}

double ComputationEngine::getVelocityAt(double x, double y, double z) const {
    if (velocityModelType == "Homogen") {
        return homogeneousVp;
    }
    else if (velocityModelType == "1D") {
        for (const auto &layer : velocityModel1D) {
            if (z <= layer.maxDepth) {
                return layer.vp;
            }
        }
        return velocityModel1D.isEmpty() ? homogeneousVp : velocityModel1D.last().vp;
    }
    else if (velocityModelType == "3D") {
        if (velocityModel3D.isEmpty()) {
            return homogeneousVp;
        }
        double lat, lon;
        kmToLatLon(x, y, lat, lon);
        double minDist = std::numeric_limits<double>::max();
        double nearestVp = homogeneousVp;
        for (const auto &point : velocityModel3D) {
            double dist = sqrt(pow(point.lat - lat, 2) + pow(point.lon - lon, 2) + pow(point.depth - z, 2));
            if (dist < minDist) {
                minDist = dist;
                nearestVp = point.vp;
            }
        }
        return nearestVp;
    }
    return homogeneousVp;
}

double ComputationEngine::calculateTravelTime(double x, double y, double z, double stationLat, double stationLon) const {
    Q_UNUSED(stationLat);
    Q_UNUSED(stationLon);
    
    // Model: distance = sqrt(x^2 + y^2 + z^2), time = distance / velocity
    // x, y, z are all in kilometers
    double dist3D = sqrt(x * x + y * y + z * z);
    
    // Get average velocity (use velocity at hypocenter depth)
    double avgVelocity = getVelocityAt(x, y, z);
    if (avgVelocity <= 0) {
        avgVelocity = 6.0;  // Default to 6 km/s if invalid
    }
    
    return dist3D / avgVelocity;
}

QVector<double> ComputationEngine::parseArrivalTimes() const {
    QVector<double> times;
    if (stations.isEmpty()) {
        return times;
    }
    QVector<QTime> arrivalTimes;
    for (const auto &station : stations) {
        QTime time = QTime::fromString(station.arrivalTime, "HH:mm:ss");
        if (!time.isValid()) {
            time = QTime::fromString(station.arrivalTime, "H:m:s");
        }
        arrivalTimes.append(time);
    }
    if (!arrivalTimes.isEmpty()) {
        QTime firstArrival = *std::min_element(arrivalTimes.begin(), arrivalTimes.end());
        for (const auto &time : arrivalTimes) {
            double seconds = firstArrival.secsTo(time);
            times.append(seconds);
        }
    }
    return times;
}

double ComputationEngine::calculateMisfit(double x, double y, double z, double t0) const {
    QVector<double> observedTimes = parseArrivalTimes();
    if (observedTimes.size() != stations.size()) {
        return std::numeric_limits<double>::max();
    }
    double sumSquaredResiduals = 0.0;
    for (int i = 0; i < stations.size(); ++i) {
        double travelTime = calculateTravelTime(x, y, z, stations[i].latitude, stations[i].longitude);
        double predicted = t0 + travelTime;
        double residual = observedTimes[i] - predicted;
        sumSquaredResiduals += residual * residual;
    }
    return sqrt(sumSquaredResiduals / stations.size());
}

QVector<double> ComputationEngine::calculateResiduals(double x, double y, double z, double t0) const {
    QVector<double> residuals;
    QVector<double> observedTimes = parseArrivalTimes();
    for (int i = 0; i < stations.size(); ++i) {
        double travelTime = calculateTravelTime(x, y, z, stations[i].latitude, stations[i].longitude);
        double predicted = t0 + travelTime;
        residuals.append(observedTimes[i] - predicted);
    }
    return residuals;
}

QVector<QVector<double>> ComputationEngine::calculateJacobian(double x, double y, double z, double t0) const {
    Q_UNUSED(t0);
    QVector<QVector<double>> J;
    const double h = 0.01;
    for (int i = 0; i < stations.size(); ++i) {
        QVector<double> row(4);
        double tt_x1 = calculateTravelTime(x + h, y, z, stations[i].latitude, stations[i].longitude);
        double tt_x0 = calculateTravelTime(x - h, y, z, stations[i].latitude, stations[i].longitude);
        row[0] = -(tt_x1 - tt_x0) / (2.0 * h);
        double tt_y1 = calculateTravelTime(x, y + h, z, stations[i].latitude, stations[i].longitude);
        double tt_y0 = calculateTravelTime(x, y - h, z, stations[i].latitude, stations[i].longitude);
        row[1] = -(tt_y1 - tt_y0) / (2.0 * h);
        double tt_z1 = calculateTravelTime(x, y, z + h, stations[i].latitude, stations[i].longitude);
        double tt_z0 = calculateTravelTime(x, y, z - h, stations[i].latitude, stations[i].longitude);
        row[2] = -(tt_z1 - tt_z0) / (2.0 * h);
        row[3] = -1.0;
        J.append(row);
    }
    return J;
}

QVector<QVector<double>> ComputationEngine::transposeMatrix(const QVector<QVector<double>> &mat) const {
    if (mat.isEmpty()) return QVector<QVector<double>>();
    int rows = mat.size();
    int cols = mat[0].size();
    QVector<QVector<double>> result(cols, QVector<double>(rows));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[j][i] = mat[i][j];
        }
    }
    return result;
}

QVector<QVector<double>> ComputationEngine::multiplyMatrices(const QVector<QVector<double>> &A, const QVector<QVector<double>> &B) const {
    int rowsA = A.size();
    int colsA = A[0].size();
    int colsB = B[0].size();
    QVector<QVector<double>> result(rowsA, QVector<double>(colsB, 0.0));
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            for (int k = 0; k < colsA; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

QVector<double> ComputationEngine::multiplyMatrixVector(const QVector<QVector<double>> &A, const QVector<double> &b) const {
    QVector<double> result(A.size(), 0.0);
    for (int i = 0; i < A.size(); ++i) {
        for (int j = 0; j < b.size(); ++j) {
            result[i] += A[i][j] * b[j];
        }
    }
    return result;
}

QVector<QVector<double>> ComputationEngine::invertMatrix(const QVector<QVector<double>> &mat) const {
    int n = mat.size();
    QVector<QVector<double>> result(n, QVector<double>(n));
    QVector<QVector<double>> temp = mat;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            result[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
    for (int i = 0; i < n; ++i) {
        double pivot = temp[i][i];
        if (fabs(pivot) < 1e-10) {
            return result;
        }
        for (int j = 0; j < n; ++j) {
            temp[i][j] /= pivot;
            result[i][j] /= pivot;
        }
        for (int k = 0; k < n; ++k) {
            if (k != i) {
                double factor = temp[k][i];
                for (int j = 0; j < n; ++j) {
                    temp[k][j] -= factor * temp[i][j];
                    result[k][j] -= factor * result[i][j];
                }
            }
        }
    }
    return result;
}

void ComputationEngine::computeGridSearch(bool useMonteCarlo, int sampleSize) {
    if (!boundarySet || !stationsSet) {
        emit computationError("Boundary or stations not set!");
        return;
    }
    emit progressUpdated(0, "Starting Grid Search (CPU)...");
    
    result.iterationNumbers.clear();
    result.misfitHistory.clear();
    
    double xMin, xMax, yMin, yMax;
    latLonToKm(boundary.yMin, boundary.xMin, xMin, yMin);
    latLonToKm(boundary.yMax, boundary.xMax, xMax, yMax);
    int nX = static_cast<int>((xMax - xMin) / boundary.gridSpacing) + 1;
    int nY = static_cast<int>((yMax - yMin) / boundary.gridSpacing) + 1;
    int nZ = static_cast<int>((boundary.depthMax - boundary.depthMin) / boundary.gridSpacing) + 1;
    long long totalPoints = static_cast<long long>(nX) * nY * nZ;
    double bestX = 0, bestY = 0, bestZ = 0, bestT0 = 0;
    double bestMisfit = std::numeric_limits<double>::max();
    long long sampleCounter = 0;
    if (useMonteCarlo && sampleSize < totalPoints) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> disX(xMin, xMax);
        std::uniform_real_distribution<> disY(yMin, yMax);
        std::uniform_real_distribution<> disZ(boundary.depthMin, boundary.depthMax);
        for (int sample = 0; sample < sampleSize; ++sample) {
            double x = disX(gen);
            double y = disY(gen);
            double z = disZ(gen);
            double t0 = 0.0;
            double misfit = calculateMisfit(x, y, z, t0);
            if (misfit < bestMisfit) {
                bestMisfit = misfit;
                bestX = x;
                bestY = y;
                bestZ = z;
                bestT0 = t0;
            }
            sampleCounter++;
            if (sample % 100 == 0) {
                result.iterationNumbers.append(sample / 100);
                result.misfitHistory.append(bestMisfit);
                int progress = static_cast<int>(100.0 * sample / sampleSize);
                emit progressUpdated(progress, QString("Monte Carlo: %1/%2 samples, best misfit = %3").arg(sample).arg(sampleSize).arg(bestMisfit, 0, 'e', 3));
            }
        }
    } else {
        long long count = 0;
        for (int ix = 0; ix < nX; ++ix) {
            for (int iy = 0; iy < nY; ++iy) {
                for (int iz = 0; iz < nZ; ++iz) {
                    double x = xMin + ix * boundary.gridSpacing;
                    double y = yMin + iy * boundary.gridSpacing;
                    double z = boundary.depthMin + iz * boundary.gridSpacing;
                    double t0 = 0.0;
                    double misfit = calculateMisfit(x, y, z, t0);
                    if (misfit < bestMisfit) {
                        bestMisfit = misfit;
                        bestX = x;
                        bestY = y;
                        bestZ = z;
                        bestT0 = t0;
                    }
                    count++;
                    if (count % 1000 == 0) {
                        result.iterationNumbers.append(count / 1000);
                        result.misfitHistory.append(bestMisfit);
                        int progress = static_cast<int>(100.0 * count / totalPoints);
                        emit progressUpdated(progress, QString("Grid Search: %1/%2 points, best misfit = %3").arg(count).arg(totalPoints).arg(bestMisfit, 0, 'e', 3));
                    }
                }
            }
        }
    }
    // Add final result if not already added
    if (result.iterationNumbers.isEmpty() || result.iterationNumbers.last() == 0) {
        result.iterationNumbers.append(useMonteCarlo ? sampleSize / 100 : static_cast<int>(totalPoints / 1000));
        result.misfitHistory.append(bestMisfit);
    }
    
    double bestLat, bestLon;
    kmToLatLon(bestX, bestY, bestLat, bestLon);
    result.x = bestLon;
    result.y = bestLat;
    result.z = bestZ;
    result.originTime = bestT0;
    result.misfit = bestMisfit;
    result.rms = bestMisfit;
    result.iterations = useMonteCarlo ? sampleSize : static_cast<int>(totalPoints);
    result.converged = true;
    result.residuals = calculateResiduals(bestX, bestY, bestZ, bestT0);
    emit progressUpdated(100, "Grid Search Complete!");
    emit computationFinished(result);
}

void ComputationEngine::computeGridSearchGPU(bool useMonteCarlo, int sampleSize) {
#ifdef CUDA_ENABLED
    if (!gpuAvailable) {
        qWarning() << "GPU not available, falling back to CPU";
        computeGridSearch(useMonteCarlo, sampleSize);
        return;
    }
    qWarning() << "GPU Grid Search under development, using CPU";
    computeGridSearch(useMonteCarlo, sampleSize);
#else
    Q_UNUSED(useMonteCarlo);
    Q_UNUSED(sampleSize);
    qWarning() << "CUDA not enabled, using CPU";
    computeGridSearch(useMonteCarlo, sampleSize);
#endif
}

void ComputationEngine::computeGaussNewtonGPU(double tolerance, int maxIter) {
#ifdef CUDA_ENABLED
    if (!gpuAvailable) {
        qWarning() << "GPU not available, falling back to CPU";
        computeGaussNewton(tolerance, maxIter);
        return;
    }
    qWarning() << "GPU Gauss-Newton under development, using CPU";
    computeGaussNewton(tolerance, maxIter);
#else
    Q_UNUSED(tolerance);
    Q_UNUSED(maxIter);
    qWarning() << "CUDA not enabled, using CPU";
    computeGaussNewton(tolerance, maxIter);
#endif
}

void ComputationEngine::computeGaussNewton(double tolerance, int maxIter) {
    if (!boundarySet || !stationsSet) {
        emit computationError("Boundary or stations not set!");
        return;
    }
    emit progressUpdated(0, "Starting Gauss-Newton...");
    double x = 0.0;
    double y = 0.0;
    double z = (boundary.depthMin + boundary.depthMax) / 2.0;
    double t0 = 0.0;  // Initialize t0 once, will accumulate during iterations
    
    result.iterationNumbers.clear();
    result.misfitHistory.clear();
    
    for (int iter = 0; iter < maxIter; ++iter) {
        QVector<double> residuals = calculateResiduals(x, y, z, t0);
        QVector<QVector<double>> J = calculateJacobian(x, y, z, t0);
        QVector<QVector<double>> JT = transposeMatrix(J);
        QVector<QVector<double>> JTJ = multiplyMatrices(JT, J);
        QVector<double> JTr = multiplyMatrixVector(JT, residuals);
        QVector<QVector<double>> JTJ_inv = invertMatrix(JTJ);
        QVector<double> delta = multiplyMatrixVector(JTJ_inv, JTr);
        
        x += delta[0];
        y += delta[1];
        z += delta[2];
        t0 += delta[3];  // Update t0 continuously
        
        double misfit = calculateMisfit(x, y, z, t0);
        result.iterationNumbers.append(iter + 1);
        result.misfitHistory.append(misfit);
        
        emit progressUpdated(static_cast<int>(100.0 * iter / maxIter), 
                           QString("Iteration %1: misfit = %2, t0 = %3").arg(iter + 1).arg(misfit, 0, 'e', 3).arg(t0, 0, 'f', 4));
    }
    
    // t0 is already calculated from iterations
    
    double lat, lon;
    kmToLatLon(x, y, lat, lon);
    result.x = lon;
    result.y = lat;
    result.z = z;
    result.originTime = t0;
    result.misfit = calculateMisfit(x, y, z, t0);
    result.rms = result.misfit;
    result.iterations = maxIter;
    result.converged = false;
    result.residuals = calculateResiduals(x, y, z, t0);
    
    emit progressUpdated(100, "Gauss-Newton: Max iterations reached");
    emit computationFinished(result);
}

void ComputationEngine::computeSteepestDescent(double tolerance, int maxIter, double stepSize) {
    if (!boundarySet || !stationsSet) {
        emit computationError("Boundary or stations not set!");
        return;
    }
    emit progressUpdated(0, "Starting Steepest Descent...");
    double x = 0.0;
    double y = 0.0;
    double z = (boundary.depthMin + boundary.depthMax) / 2.0;
    double t0 = 0.0;  // Initialize t0 once, will accumulate during iterations
    
    result.iterationNumbers.clear();
    result.misfitHistory.clear();
    
    for (int iter = 0; iter < maxIter; ++iter) {
        QVector<double> residuals = calculateResiduals(x, y, z, t0);
        QVector<QVector<double>> J = calculateJacobian(x, y, z, t0);
        QVector<QVector<double>> JT = transposeMatrix(J);
        QVector<double> gradient = multiplyMatrixVector(JT, residuals);
        
        double normG = std::sqrt(std::accumulate(gradient.begin(), gradient.end(), 0.0,
            [](double sum, double g) { return sum + g*g; }));
        if (normG < 1e-10) break;
        
        // Normalize gradient
        for (auto &g : gradient) g /= normG;
        
        x -= stepSize * gradient[0];
        y -= stepSize * gradient[1];
        z -= stepSize * gradient[2];
        t0 -= stepSize * gradient[3];  // Update t0 continuously
        
        double misfit = calculateMisfit(x, y, z, t0);
        result.iterationNumbers.append(iter + 1);
        result.misfitHistory.append(misfit);
        
        emit progressUpdated(static_cast<int>(100.0 * iter / maxIter), 
                           QString("Iteration %1: misfit = %2, t0 = %3").arg(iter + 1).arg(misfit, 0, 'e', 3).arg(t0, 0, 'f', 4));
    }
    
    double lat, lon;
    kmToLatLon(x, y, lat, lon);
    result.x = lon;
    result.y = lat;
    result.z = z;
    result.originTime = t0;
    result.misfit = calculateMisfit(x, y, z, t0);
    result.rms = result.misfit;
    result.iterations = maxIter;
    result.converged = false;
    result.residuals = calculateResiduals(x, y, z, t0);
    
    emit progressUpdated(100, "Steepest Descent: Max iterations reached");
    emit computationFinished(result);
}

void ComputationEngine::computeLevenbergMarquardt(double tolerance, int maxIter, double lambda) {
    if (!boundarySet || !stationsSet) {
        emit computationError("Boundary or stations not set!");
        return;
    }
    emit progressUpdated(0, "Starting Levenberg-Marquardt...");
    double x = 0.0;
    double y = 0.0;
    double z = (boundary.depthMin + boundary.depthMax) / 2.0;
    double t0 = 0.0;  // Initialize t0 once, will accumulate during iterations
    
    result.iterationNumbers.clear();
    result.misfitHistory.clear();
    
    double currentLambda = lambda;
    for (int iter = 0; iter < maxIter; ++iter) {
        QVector<double> residuals = calculateResiduals(x, y, z, t0);
        QVector<QVector<double>> J = calculateJacobian(x, y, z, t0);
        QVector<QVector<double>> JT = transposeMatrix(J);
        QVector<QVector<double>> JTJ = multiplyMatrices(JT, J);
        
        for (int i = 0; i < JTJ.size(); ++i) {
            JTJ[i][i] += currentLambda;
        }
        
        QVector<double> JTr = multiplyMatrixVector(JT, residuals);
        QVector<QVector<double>> JTJ_inv = invertMatrix(JTJ);
        QVector<double> delta = multiplyMatrixVector(JTJ_inv, JTr);
        
        double newX = x + delta[0];
        double newY = y + delta[1];
        double newZ = z + delta[2];
        double newT0 = t0 + delta[3];  // Update t0
        
        double oldMisfit = calculateMisfit(x, y, z, t0);
        double newMisfit = calculateMisfit(newX, newY, newZ, newT0);
        
        if (newMisfit < oldMisfit) {
            x = newX;
            y = newY;
            z = newZ;
            t0 = newT0;  // Commit t0 update
            currentLambda *= 0.1;
        } else {
            currentLambda *= 10.0;
        }
        
        result.iterationNumbers.append(iter + 1);
        result.misfitHistory.append(std::min(oldMisfit, newMisfit));
        
        emit progressUpdated(static_cast<int>(100.0 * iter / maxIter), 
                           QString("Iteration %1: misfit = %2, t0 = %3, lambda = %4")
                           .arg(iter + 1).arg(newMisfit, 0, 'e', 3).arg(t0, 0, 'f', 4).arg(currentLambda, 0, 'e', 2));
    }
    
    double lat, lon;
    kmToLatLon(x, y, lat, lon);
    result.x = lon;
    result.y = lat;
    result.z = z;
    result.originTime = t0;
    result.misfit = calculateMisfit(x, y, z, t0);
    result.rms = result.misfit;
    result.iterations = maxIter;
    result.converged = false;
    result.residuals = calculateResiduals(x, y, z, t0);
    
    emit progressUpdated(100, "Levenberg-Marquardt: Max iterations reached");
    emit computationFinished(result);
}
