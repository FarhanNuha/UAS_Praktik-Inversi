#ifndef SHAREDTYPES_H
#define SHAREDTYPES_H

#include <QString>
#include <QVector>
#include <QPointF>

// Boundary data for calculation area
struct BoundaryData {
    double xMin, xMax;      // Longitude range
    double yMin, yMax;      // Latitude range
    double depthMin, depthMax;  // Depth range (km)
    double gridSpacing;     // Grid spacing (km)
};

// Station data for seismic stations
struct StationData {
    QString name;
    double latitude;
    double longitude;
    QString arrivalTime;    // Format: "HH:mm:ss"
};

// 1D velocity model layer
struct VelocityLayer1D {
    double vp;              // P-wave velocity (km/s)
    double maxDepth;        // Maximum depth of this layer (km)
};

// 3D velocity model point
struct VelocityPoint3D {
    double lat, lon, depth;
    double vp;              // P-wave velocity (km/s)
};

// Hypocenter result from inversion
struct HypocenterResult {
    double x;          // Longitude (degrees)
    double y;          // Latitude (degrees)
    double z;          // Depth (km)
    double originTime; // Origin time in seconds from first arrival
    double misfit;     // RMS residual (seconds)
    double rms;        // Root mean square error
    int iterations;    // Number of iterations performed
    bool converged;    // Convergence status
    
    QVector<double> residuals;      // Per-station residuals
    QVector<double> travelTimes;    // Calculated travel times
    QVector<double> observedTimes;  // Observed arrival times
    
    // For iteration tracking
    QVector<double> iterationNumbers;
    QVector<double> misfitHistory;
    
    // For visualization
    QVector<QPointF> contour2D;
};

// Method configuration
struct MethodData {
    QString approach;
    QString methodName;
    bool useMonteCarloSampling;
    int monteCarloSamples;
};

// Velocity model configuration
struct VelocityData {
    QString modelType;
    double homogeneousVp;
    QVector<VelocityLayer1D> layers1D;
    QVector<VelocityPoint3D> points3D;
};

#endif // SHAREDTYPES_H
