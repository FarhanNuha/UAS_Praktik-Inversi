#include "InversionMethods.h"
#include <algorithm>
#include <chrono>
#include <iostream>

// =============================================================================
// GENETIC ALGORITHM METHOD
// =============================================================================

GeneticAlgorithmMethod::GeneticAlgorithmMethod(
    GAVariant variant,
    int popSize, int maxGen,
    double crossoverRate, double mutationRate,
    int eliteSize, bool realCoded)
    : variant(variant), populationSize(popSize), maxGenerations(maxGen),
      crossoverRate(crossoverRate), mutationRate(mutationRate),
      eliteSize(eliteSize), realCoded(realCoded)
{
    std::random_device rd;
    rng.seed(rd());
}

void GeneticAlgorithmMethod::initializePopulation(
    std::vector<Individual>& pop,
    const SearchBoundary& boundary)
{
    pop.resize(populationSize);
    
    std::uniform_real_distribution<double> distX(boundary.xMin, boundary.xMax);
    std::uniform_real_distribution<double> distY(boundary.yMin, boundary.yMax);
    std::uniform_real_distribution<double> distZ(boundary.zMin, boundary.zMax);
    std::uniform_real_distribution<double> distT0(-100, 100);
    
    for (auto& ind : pop) {
        ind.x = distX(rng);
        ind.y = distY(rng);
        ind.z = distZ(rng);
        ind.t0 = distT0(rng);
    }
}

void GeneticAlgorithmMethod::evaluateFitness(
    std::vector<Individual>& pop,
    const std::vector<Station>& stations,
    const VelocityModel* vm)
{
    for (auto& ind : pop) {
        ind.fitness = computeMisfit(ind.x, ind.y, ind.z, ind.t0, stations, vm);
    }
}

void GeneticAlgorithmMethod::selection(
    const std::vector<Individual>& pop,
    std::vector<Individual>& parents)
{
    // Tournament selection
    int tournamentSize = 5;
    parents.clear();
    
    std::uniform_int_distribution<int> dist(0, pop.size() - 1);
    
    for (size_t i = 0; i < pop.size(); ++i) {
        // Select tournament candidates
        std::vector<const Individual*> tournament;
        for (int j = 0; j < tournamentSize; ++j) {
            tournament.push_back(&pop[dist(rng)]);
        }
        
        // Find best in tournament
        const Individual* best = tournament[0];
        for (const auto* ind : tournament) {
            if (ind->fitness < best->fitness) {
                best = ind;
            }
        }
        
        parents.push_back(*best);
    }
}

void GeneticAlgorithmMethod::crossover(
    const Individual& parent1, const Individual& parent2,
    Individual& offspring1, Individual& offspring2,
    const SearchBoundary& boundary)
{
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    if (dist(rng) < crossoverRate) {
        if (realCoded) {
            // Simulated Binary Crossover (SBX)
            double eta = 2.0; // Distribution index
            
            double u = dist(rng);
            double beta = (u <= 0.5) ? pow(2*u, 1.0/(eta+1)) : pow(1.0/(2*(1-u)), 1.0/(eta+1));
            
            offspring1.x = 0.5 * ((1 + beta) * parent1.x + (1 - beta) * parent2.x);
            offspring1.y = 0.5 * ((1 + beta) * parent1.y + (1 - beta) * parent2.y);
            offspring1.z = 0.5 * ((1 + beta) * parent1.z + (1 - beta) * parent2.z);
            offspring1.t0 = 0.5 * ((1 + beta) * parent1.t0 + (1 - beta) * parent2.t0);
            
            offspring2.x = 0.5 * ((1 - beta) * parent1.x + (1 + beta) * parent2.x);
            offspring2.y = 0.5 * ((1 - beta) * parent1.y + (1 + beta) * parent2.y);
            offspring2.z = 0.5 * ((1 - beta) * parent1.z + (1 + beta) * parent2.z);
            offspring2.t0 = 0.5 * ((1 - beta) * parent1.t0 + (1 + beta) * parent2.t0);
            
            // Apply boundary constraints
            offspring1.x = std::max(boundary.xMin, std::min(boundary.xMax, offspring1.x));
            offspring1.y = std::max(boundary.yMin, std::min(boundary.yMax, offspring1.y));
            offspring1.z = std::max(boundary.zMin, std::min(boundary.zMax, offspring1.z));
            
            offspring2.x = std::max(boundary.xMin, std::min(boundary.xMax, offspring2.x));
            offspring2.y = std::max(boundary.yMin, std::min(boundary.yMax, offspring2.y));
            offspring2.z = std::max(boundary.zMin, std::min(boundary.zMax, offspring2.z));
        } else {
            // Single-point crossover for binary representation
            // (simplified - assuming real-coded for this implementation)
            double alpha = dist(rng);
            
            offspring1.x = alpha * parent1.x + (1 - alpha) * parent2.x;
            offspring1.y = alpha * parent1.y + (1 - alpha) * parent2.y;
            offspring1.z = alpha * parent1.z + (1 - alpha) * parent2.z;
            offspring1.t0 = alpha * parent1.t0 + (1 - alpha) * parent2.t0;
            
            offspring2.x = (1 - alpha) * parent1.x + alpha * parent2.x;
            offspring2.y = (1 - alpha) * parent1.y + alpha * parent2.y;
            offspring2.z = (1 - alpha) * parent1.z + alpha * parent2.z;
            offspring2.t0 = (1 - alpha) * parent1.t0 + alpha * parent2.t0;
        }
    } else {
        // No crossover - copy parents
        offspring1 = parent1;
        offspring2 = parent2;
    }
}

void GeneticAlgorithmMethod::mutate(
    Individual& ind,
    const SearchBoundary& boundary)
{
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::normal_distribution<double> mutDist(0.0, 0.1);
    
    if (dist(rng) < mutationRate) {
        // Gaussian mutation
        ind.x += mutDist(rng) * (boundary.xMax - boundary.xMin);
        ind.y += mutDist(rng) * (boundary.yMax - boundary.yMin);
        ind.z += mutDist(rng) * (boundary.zMax - boundary.zMin);
        ind.t0 += mutDist(rng) * 10.0;
        
        // Apply boundary constraints
        ind.x = std::max(boundary.xMin, std::min(boundary.xMax, ind.x));
        ind.y = std::max(boundary.yMin, std::min(boundary.yMax, ind.y));
        ind.z = std::max(boundary.zMin, std::min(boundary.zMax, ind.z));
    }
}

Individual GeneticAlgorithmMethod::findBest(const std::vector<Individual>& pop) const {
    Individual best = pop[0];
    for (const auto& ind : pop) {
        if (ind.fitness < best.fitness) {
            best = ind;
        }
    }
    return best;
}

InversionResult GeneticAlgorithmMethod::solve(
    const std::vector<Station>& stations,
    const SearchBoundary& boundary,
    const VelocityModel* velocityModel)
{
    auto startTime = std::chrono::high_resolution_clock::now();
    
    InversionResult result;
    result.converged = false;
    
    std::cout << "Starting Genetic Algorithm..." << std::endl;
    std::cout << "Variant: ";
    switch(variant) {
        case GAVariant::Standard: std::cout << "Standard GA"; break;
        case GAVariant::SteadyState: std::cout << "Steady State GA"; break;
        case GAVariant::SPEA2: std::cout << "SPEA2"; break;
    }
    std::cout << std::endl;
    
    // Initialize population
    std::vector<Individual> population;
    initializePopulation(population, boundary);
    evaluateFitness(population, stations, velocityModel);
    
    Individual bestEver = findBest(population);
    
    if (variant == GAVariant::Standard) {
        // ===== STANDARD GENETIC ALGORITHM =====
        for (int gen = 0; gen < maxGenerations; ++gen) {
            // Selection
            std::vector<Individual> parents;
            selection(population, parents);
            
            // Crossover and mutation
            std::vector<Individual> offspring;
            for (size_t i = 0; i < parents.size(); i += 2) {
                if (i + 1 < parents.size()) {
                    Individual child1, child2;
                    crossover(parents[i], parents[i+1], child1, child2, boundary);
                    mutate(child1, boundary);
                    mutate(child2, boundary);
                    offspring.push_back(child1);
                    offspring.push_back(child2);
                }
            }
            
            // Evaluate offspring
            evaluateFitness(offspring, stations, velocityModel);
            
            // Elitism: keep best individuals
            std::sort(population.begin(), population.end(),
                     [](const Individual& a, const Individual& b) {
                         return a.fitness < b.fitness;
                     });
            
            // Replace population (keep elite)
            for (int i = eliteSize; i < populationSize && i - eliteSize < (int)offspring.size(); ++i) {
                population[i] = offspring[i - eliteSize];
            }
            
            // Update best
            Individual currentBest = findBest(population);
            if (currentBest.fitness < bestEver.fitness) {
                bestEver = currentBest;
            }
            
            result.misfitHistory.push_back(bestEver.fitness);
            
            if ((gen + 1) % 10 == 0) {
                std::cout << "Generation " << (gen + 1) << "/" << maxGenerations
                         << ": best fitness = " << bestEver.fitness << std::endl;
            }
        }
        
    } else if (variant == GAVariant::SteadyState) {
        // ===== STEADY STATE GENETIC ALGORITHM =====
        int maxIterations = maxGenerations * populationSize;
        
        for (int iter = 0; iter < maxIterations; ++iter) {
            // Select 2 parents
            std::vector<Individual> parents;
            selection(population, parents);
            
            // Generate 2 offspring
            Individual child1, child2;
            crossover(parents[0], parents[1], child1, child2, boundary);
            mutate(child1, boundary);
            mutate(child2, boundary);
            
            // Evaluate offspring
            child1.fitness = computeMisfit(child1.x, child1.y, child1.z, child1.t0, stations, velocityModel);
            child2.fitness = computeMisfit(child2.x, child2.y, child2.z, child2.t0, stations, velocityModel);
            
            // Replace worst individuals
            std::sort(population.begin(), population.end(),
                     [](const Individual& a, const Individual& b) {
                         return a.fitness < b.fitness;
                     });
            
            if (child1.fitness < population.back().fitness) {
                population.back() = child1;
            }
            if (child2.fitness < population[populationSize-2].fitness) {
                population[populationSize-2] = child2;
            }
            
            // Update best
            Individual currentBest = findBest(population);
            if (currentBest.fitness < bestEver.fitness) {
                bestEver = currentBest;
            }
            
            if ((iter + 1) % (populationSize * 10) == 0) {
                result.misfitHistory.push_back(bestEver.fitness);
                std::cout << "Iteration " << (iter + 1) << "/" << maxIterations
                         << ": best fitness = " << bestEver.fitness << std::endl;
            }
        }
        
    } else if (variant == GAVariant::SPEA2) {
        // ===== SPEA2 (Simplified) =====
        // For single-objective, SPEA2 reduces to standard GA with archive
        
        std::vector<Individual> archive;
        int archiveSize = populationSize;
        
        for (int gen = 0; gen < maxGenerations; ++gen) {
            // Combine population and archive
            std::vector<Individual> combined = population;
            combined.insert(combined.end(), archive.begin(), archive.end());
            
            // Sort by fitness
            std::sort(combined.begin(), combined.end(),
                     [](const Individual& a, const Individual& b) {
                         return a.fitness < b.fitness;
                     });
            
            // Update archive (best individuals)
            archive.clear();
            for (int i = 0; i < archiveSize && i < (int)combined.size(); ++i) {
                archive.push_back(combined[i]);
            }
            
            // Selection from archive
            std::vector<Individual> parents;
            selection(archive, parents);
            
            // Generate offspring
            std::vector<Individual> offspring;
            for (size_t i = 0; i < parents.size(); i += 2) {
                if (i + 1 < parents.size()) {
                    Individual child1, child2;
                    crossover(parents[i], parents[i+1], child1, child2, boundary);
                    mutate(child1, boundary);
                    mutate(child2, boundary);
                    offspring.push_back(child1);
                    offspring.push_back(child2);
                }
            }
            
            // Evaluate offspring
            evaluateFitness(offspring, stations, velocityModel);
            
            // Replace population
            population = offspring;
            if (population.size() > (size_t)populationSize) {
                population.resize(populationSize);
            }
            
            // Update best
            Individual currentBest = findBest(archive);
            if (currentBest.fitness < bestEver.fitness) {
                bestEver = currentBest;
            }
            
            result.misfitHistory.push_back(bestEver.fitness);
            
            if ((gen + 1) % 10 == 0) {
                std::cout << "Generation " << (gen + 1) << "/" << maxGenerations
                         << ": best fitness = " << bestEver.fitness << std::endl;
            }
        }
    }
    
    // Set result
    result.x = bestEver.x;
    result.y = bestEver.y;
    result.z = bestEver.z;
    result.originTime = bestEver.t0;
    result.misfit = bestEver.fitness;
    result.iterations = maxGenerations;
    result.converged = true;
    
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;
    result.computationTime = elapsed.count();
    
    std::cout << "Genetic Algorithm completed!" << std::endl;
    std::cout << "Best location: (" << result.x << ", " << result.y << ", " << result.z << ")" << std::endl;
    std::cout << "Best misfit: " << result.misfit << std::endl;
    
    return result;
}
