#ifndef COMPUTATIONENGINE_H
#define COMPUTATIONENGINE_H

#include <QObject>
#include <QVector>
#include <QDateTime>
#include <QString>
#include <cmath>
#include <limits>
#include <random>
#include "SharedTypes.h"

class ComputationEngine : public QObject {
    Q_OBJECT
    
public:
    explicit ComputationEngine(QObject *parent = nullptr);
    ~ComputationEngine();
    
    // Setup functions
    void setBoundary(const BoundaryData &boundary);
    void setStations(const QVector<StationData> &stations);
    void setVelocityModel(const QString &type, double homogeneousVp = 6.0);
    void setVelocityModel1D(const QVector<VelocityLayer1D> &layers);
    void setVelocityModel3D(const QVector<VelocityPoint3D> &points);
    
    // CPU Computation methods
    void computeGridSearch(bool useMonteCarlo, int sampleSize);
    void computeGaussNewton(double tolerance, int maxIter);
    void computeSteepestDescent(double tolerance, int maxIter, double stepSize);
    void computeLevenbergMarquardt(double tolerance, int maxIter, double lambda);
    
    // GPU Computation methods
    void computeGridSearchGPU(bool useMonteCarlo, int sampleSize);
    void computeGaussNewtonGPU(double tolerance, int maxIter);
    
    HypocenterResult getResult() const { return result; }
    
    // Check GPU availability
    bool isGPUAvailable() const;
    QString getGPUInfo() const;
    
signals:
    void progressUpdated(int percent, const QString &message);
    void computationFinished(const HypocenterResult &result);
    void computationError(const QString &error);
    
private:
    // Helper functions
    double calculateTravelTime(double x, double y, double z, 
                              double stationLat, double stationLon) const;
    double getVelocityAt(double x, double y, double z) const;
    double calculateMisfit(double x, double y, double z, double t0) const;
    void latLonToKm(double lat, double lon, double &x, double &y) const;
    void kmToLatLon(double x, double y, double &lat, double &lon) const;
    double haversineDistance(double lat1, double lon1, double lat2, double lon2) const;
    QVector<double> parseArrivalTimes() const;
    
    // Numerical methods helpers
    QVector<double> calculateResiduals(double x, double y, double z, double t0) const;
    QVector<QVector<double>> calculateJacobian(double x, double y, double z, double t0) const;
    QVector<QVector<double>> transposeMatrix(const QVector<QVector<double>> &mat) const;
    QVector<QVector<double>> multiplyMatrices(const QVector<QVector<double>> &A,
                                             const QVector<QVector<double>> &B) const;
    QVector<double> multiplyMatrixVector(const QVector<QVector<double>> &A,
                                        const QVector<double> &b) const;
    QVector<QVector<double>> invertMatrix(const QVector<QVector<double>> &mat) const;
    
    // GPU helper functions
    void initializeGPU();
    void cleanupGPU();
    
    // Data members
    BoundaryData boundary;
    QVector<StationData> stations;
    bool boundarySet;
    bool stationsSet;
    
    // Velocity model
    QString velocityModelType;
    double homogeneousVp;
    QVector<VelocityLayer1D> velocityModel1D;
    QVector<VelocityPoint3D> velocityModel3D;
    
    // Result
    HypocenterResult result;
    
    // Reference point for coordinate conversion
    double refLat, refLon;
    
    // GPU state
    bool gpuInitialized;
    bool gpuAvailable;
    QString gpuDeviceName;
};

#endif // COMPUTATIONENGINE_H
