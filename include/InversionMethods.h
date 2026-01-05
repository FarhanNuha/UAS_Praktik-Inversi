#ifndef INVERSIONMETHODS_H
#define INVERSIONMETHODS_H

#include <vector>
#include <functional>
#include <random>

// Structure untuk hasil inversi
struct InversionResult {
    double x, y, z;           // Lokasi hiposenter
    double originTime;        // Origin time
    double misfit;            // Final misfit
    int iterations;           // Jumlah iterasi
    std::vector<double> misfitHistory;  // History misfit per iterasi
    bool converged;           // Status konvergensi
    double computationTime;   // Waktu komputasi (detik)
};

// Structure untuk station data
struct Station {
    double lat, lon;
    double arrivalTime;
    std::string name;
};

// Structure untuk boundary
struct SearchBoundary {
    double xMin, xMax;
    double yMin, yMax;
    double zMin, zMax;
    double gridSpacing;
};

// Forward declaration untuk velocity model
class VelocityModel;

// =============================================================================
// LOCAL METHODS
// =============================================================================

class LocalInversionMethod {
public:
    virtual ~LocalInversionMethod() = default;
    
    virtual InversionResult solve(
        const std::vector<Station>& stations,
        const SearchBoundary& boundary,
        const VelocityModel* velocityModel,
        double initialX, double initialY, double initialZ,
        double initialOriginTime
    ) = 0;
    
protected:
    // Helper: compute travel time from source to station
    double computeTravelTime(double srcX, double srcY, double srcZ,
                            double staLat, double staLon,
                            const VelocityModel* vm) const;
    
    // Helper: compute misfit (residual)
    double computeMisfit(double x, double y, double z, double t0,
                        const std::vector<Station>& stations,
                        const VelocityModel* vm) const;
    
    // Helper: compute Jacobian matrix
    void computeJacobian(double x, double y, double z, double t0,
                        const std::vector<Station>& stations,
                        const VelocityModel* vm,
                        std::vector<std::vector<double>>& J) const;
};

// Gauss-Newton Method
class GaussNewtonMethod : public LocalInversionMethod {
public:
    GaussNewtonMethod(int maxIter = 100, double tol = 1e-6, double damping = 0.001);
    
    InversionResult solve(
        const std::vector<Station>& stations,
        const SearchBoundary& boundary,
        const VelocityModel* velocityModel,
        double initialX, double initialY, double initialZ,
        double initialOriginTime
    ) override;
    
private:
    int maxIterations;
    double tolerance;
    double dampingFactor;
};

// Steepest Descent Method
class SteepestDescentMethod : public LocalInversionMethod {
public:
    SteepestDescentMethod(int maxIter = 200, double tol = 1e-5, 
                         double stepSize = 0.01, int lineSearchIter = 10);
    
    InversionResult solve(
        const std::vector<Station>& stations,
        const SearchBoundary& boundary,
        const VelocityModel* velocityModel,
        double initialX, double initialY, double initialZ,
        double initialOriginTime
    ) override;
    
private:
    int maxIterations;
    double tolerance;
    double initialStepSize;
    int lineSearchIterations;
};

// Levenberg-Marquardt Method
class LevenbergMarquardtMethod : public LocalInversionMethod {
public:
    LevenbergMarquardtMethod(int maxIter = 100, double tol = 1e-6,
                            double lambda = 0.01, double lambdaUp = 10.0,
                            double lambdaDown = 0.1);
    
    InversionResult solve(
        const std::vector<Station>& stations,
        const SearchBoundary& boundary,
        const VelocityModel* velocityModel,
        double initialX, double initialY, double initialZ,
        double initialOriginTime
    ) override;
    
private:
    int maxIterations;
    double tolerance;
    double lambda;
    double lambdaUpFactor;
    double lambdaDownFactor;
};

// =============================================================================
// GLOBAL METHODS
// =============================================================================

class GlobalInversionMethod {
public:
    virtual ~GlobalInversionMethod() = default;
    
    virtual InversionResult solve(
        const std::vector<Station>& stations,
        const SearchBoundary& boundary,
        const VelocityModel* velocityModel
    ) = 0;
    
protected:
    double computeMisfit(double x, double y, double z, double t0,
                        const std::vector<Station>& stations,
                        const VelocityModel* vm) const;
    
    double computeTravelTime(double srcX, double srcY, double srcZ,
                            double staLat, double staLon,
                            const VelocityModel* vm) const;
};

// Grid Search Method
class GridSearchMethod : public GlobalInversionMethod {
public:
    GridSearchMethod(bool useMonteCarlo = false, int sampleSize = 1000);
    
    InversionResult solve(
        const std::vector<Station>& stations,
        const SearchBoundary& boundary,
        const VelocityModel* velocityModel
    ) override;
    
private:
    bool monteCarloSampling;
    int numSamples;
    std::mt19937 rng;
};

// Simulated Annealing Method
enum class SAVariant {
    Simple,
    Metropolis,
    Cauchy
};

enum class CoolingSchedule {
    Exponential,
    Linear,
    Logarithmic,
    Inverse,
    Adaptive,
    CauchySchedule,
    FastCauchy,
    VeryFast
};

class SimulatedAnnealingMethod : public GlobalInversionMethod {
public:
    SimulatedAnnealingMethod(
        SAVariant variant = SAVariant::Simple,
        double T0 = 1000.0,
        double Tf = 0.1,
        CoolingSchedule schedule = CoolingSchedule::Exponential,
        double alpha = 0.95,
        int iterPerTemp = 100,
        int maxIter = 10000
    );
    
    InversionResult solve(
        const std::vector<Station>& stations,
        const SearchBoundary& boundary,
        const VelocityModel* velocityModel
    ) override;
    
private:
    SAVariant variant;
    double initialTemp;
    double finalTemp;
    CoolingSchedule coolingSchedule;
    double coolingAlpha;
    int iterationsPerTemp;
    int maxIterations;
    std::mt19937 rng;
    
    double updateTemperature(double T, int iteration) const;
    bool acceptSolution(double currentMisfit, double newMisfit, double T) const;
    void generateNeighbor(double& x, double& y, double& z, double& t0,
                         const SearchBoundary& boundary, double T) const;
};

// Genetic Algorithm Method
enum class GAVariant {
    Standard,
    SteadyState,
    SPEA2
};

struct Individual {
    double x, y, z, t0;
    double fitness;
    
    Individual() : x(0), y(0), z(0), t0(0), fitness(1e9) {}
    Individual(double x_, double y_, double z_, double t0_) 
        : x(x_), y(y_), z(z_), t0(t0_), fitness(1e9) {}
};

class GeneticAlgorithmMethod : public GlobalInversionMethod {
public:
    GeneticAlgorithmMethod(
        GAVariant variant = GAVariant::Standard,
        int popSize = 100,
        int maxGen = 200,
        double crossoverRate = 0.8,
        double mutationRate = 0.1,
        int eliteSize = 10,
        bool realCoded = true
    );
    
    InversionResult solve(
        const std::vector<Station>& stations,
        const SearchBoundary& boundary,
        const VelocityModel* velocityModel
    ) override;
    
private:
    GAVariant variant;
    int populationSize;
    int maxGenerations;
    double crossoverRate;
    double mutationRate;
    int eliteSize;
    bool realCoded;
    std::mt19937 rng;
    
    void initializePopulation(std::vector<Individual>& pop,
                             const SearchBoundary& boundary);
    
    void evaluateFitness(std::vector<Individual>& pop,
                        const std::vector<Station>& stations,
                        const VelocityModel* vm);
    
    void selection(const std::vector<Individual>& pop,
                  std::vector<Individual>& parents);
    
    void crossover(const Individual& parent1, const Individual& parent2,
                  Individual& offspring1, Individual& offspring2,
                  const SearchBoundary& boundary);
    
    void mutate(Individual& ind, const SearchBoundary& boundary);
    
    Individual findBest(const std::vector<Individual>& pop) const;
};

// =============================================================================
// VELOCITY MODEL INTERFACE
// =============================================================================

class VelocityModel {
public:
    virtual ~VelocityModel() = default;
    virtual double getVelocity(double lat, double lon, double depth) const = 0;
};

class HomogeneousVelocityModel : public VelocityModel {
public:
    HomogeneousVelocityModel(double vp) : velocity(vp) {}
    double getVelocity(double lat, double lon, double depth) const override {
        (void)lat; (void)lon; (void)depth;
        return velocity;
    }
private:
    double velocity;
};

class OneDVelocityModel : public VelocityModel {
public:
    struct Layer {
        double vp;
        double maxDepth;
    };
    
    OneDVelocityModel(const std::vector<Layer>& layers) : layers(layers) {}
    
    double getVelocity(double lat, double lon, double depth) const override {
        (void)lat; (void)lon;
        for (const auto& layer : layers) {
            if (depth <= layer.maxDepth) {
                return layer.vp;
            }
        }
        return layers.back().vp;
    }
    
private:
    std::vector<Layer> layers;
};

class ThreeDVelocityModel : public VelocityModel {
public:
    struct GridPoint {
        double lat, lon, depth, vp;
    };
    
    ThreeDVelocityModel(const std::vector<GridPoint>& grid,
                       const SearchBoundary& boundary);
    
    double getVelocity(double lat, double lon, double depth) const override;
    
private:
    std::vector<GridPoint> gridData;
    SearchBoundary bounds;
    
    // Trilinear interpolation
    double interpolate(double lat, double lon, double depth) const;
};

#endif // INVERSIONMETHODS_H
