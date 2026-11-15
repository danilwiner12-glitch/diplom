#include <H5Cpp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Field3D.hpp"
#include "Geometry3D.hpp"
#include "Hdf5Writer.hpp"
#include "Solver3D.hpp"
#include "YeeGrid3D.hpp"

namespace {
constexpr double c0 = 299792458.0;
constexpr double electronCharge = -1.602176634e-19;
constexpr double electronMass = 9.1093837015e-31;

struct RunConfiguration {
    std::size_t nx = 96;
    std::size_t ny = 48;
    std::size_t nz = 48;
    std::size_t steps = 180;
    std::size_t outputEvery = 15;
    std::string outputPath = "cherenkov.h5";
};

std::size_t parsePositiveSize(const char* text, const std::string& option) {
    const std::string value(text);
    std::size_t parsedCharacters = 0;
    const unsigned long long parsed = std::stoull(value, &parsedCharacters);
    if (parsedCharacters != value.size() || parsed == 0) {
        throw std::runtime_error(option + " expects a positive integer");
    }
    return static_cast<std::size_t>(parsed);
}

RunConfiguration parseArguments(int argc, char** argv) {
    RunConfiguration configuration;
    for (int argument = 1; argument < argc; ++argument) {
        const std::string option(argv[argument]);
        if (option == "--quick") {
            configuration.nx = 32;
            configuration.ny = 20;
            configuration.nz = 20;
            configuration.steps = 20;
            configuration.outputEvery = 5;
        } else if (option == "--steps" && argument + 1 < argc) {
            configuration.steps =
                parsePositiveSize(argv[++argument], option);
        } else if (option == "--output-every" && argument + 1 < argc) {
            configuration.outputEvery =
                parsePositiveSize(argv[++argument], option);
        } else if (option == "--output" && argument + 1 < argc) {
            configuration.outputPath = argv[++argument];
        } else if (option == "--help") {
            std::cout
                << "Usage: pic3d [--quick] [--steps N] "
                   "[--output-every N] [--output FILE]\n";
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::runtime_error("Unknown or incomplete option: " + option);
        }
    }
    return configuration;
}

double maximumMagnitude(const std::vector<double>& values) {
    double maximum = 0.0;
    for (const double value : values) {
        maximum = std::max(maximum, std::abs(value));
    }
    return maximum;
}
}

int main(int argc, char** argv) {
    try {
        const RunConfiguration configuration = parseArguments(argc, argv);
        const double cellSize = 1.0e-3;
        const double lx = cellSize * static_cast<double>(configuration.nx);
        const double ly = cellSize * static_cast<double>(configuration.ny);
        const double lz = cellSize * static_cast<double>(configuration.nz);
        const YeeGrid3D grid(configuration.nx, configuration.ny,
                             configuration.nz, lx, ly, lz);

        const double cflLimit = 1.0 /
            (c0 * std::sqrt(grid.invDx * grid.invDx +
                            grid.invDy * grid.invDy +
                            grid.invDz * grid.invDz));
        const double dt = 0.95 * cflLimit;

        const Material dielectric("uniform dielectric", 2.25, 1.0, 0.0);
        const Geometry3D geometry(dielectric);
        Field3D field(grid);
        Solver3D solver(grid, field, dt, geometry,
                        ParticleBoundaryCondition::Absorb);

        const double beta = 0.99;
        const double velocity = beta * c0;
        const double gamma = 1.0 / std::sqrt(1.0 - beta * beta);
        const double macroParticleWeight = 1.0e6;
        const double charge = macroParticleWeight * electronCharge;
        const double mass = macroParticleWeight * electronMass;
        const double momentumX = gamma * mass * velocity;
        solver.addParticle(8.0 * grid.dx, 0.5 * ly, 0.5 * lz,
                           momentumX, 0.0, 0.0, charge, mass);
        solver.initializeSources();

        const double refractiveIndex = dielectric.refractiveIndex();
        const double thresholdBeta = 1.0 / refractiveIndex;
        if (beta <= thresholdBeta) {
            throw std::runtime_error(
                "Particle velocity is below the Cherenkov threshold");
        }
        const double expectedAngle =
            std::acos(1.0 / (refractiveIndex * beta));
        const double kineticEnergyEv =
            (gamma - 1.0) * electronMass * c0 * c0 /
            std::abs(electronCharge);

        std::cout << "Uniform dielectric: epsilon_r = "
                  << dielectric.relativePermittivity
                  << ", n = " << refractiveIndex << '\n'
                  << "Electron: beta = " << beta
                  << ", kinetic energy = " << kineticEnergyEv / 1.0e6
                  << " MeV\n"
                  << "Cherenkov threshold beta > " << thresholdBeta
                  << ", expected angle = "
                  << expectedAngle * 180.0 / std::acos(-1.0)
                  << " degrees\n"
                  << "Grid: " << grid.nx << " x " << grid.ny << " x "
                  << grid.nz << ", dt = " << dt << " s\n";

        Hdf5Writer writer(configuration.outputPath, grid, geometry);
        writer.writeFrame(0, solver.time, field, solver.particles);

        for (std::size_t step = 1; step <= configuration.steps; ++step) {
            solver.step();
            if (step % configuration.outputEvery == 0 ||
                step == configuration.steps) {
                writer.writeFrame(step, solver.time, field, solver.particles);
                const double maxElectric = std::max(
                    {maximumMagnitude(field.Ex),
                     maximumMagnitude(field.Ey),
                     maximumMagnitude(field.Ez)});
                std::cout << "step " << step << '/' << configuration.steps
                          << ", particles = " << solver.particles.size()
                          << ", max|E component| = " << maxElectric
                          << " V/m\n";
            }
        }

        std::cout << "HDF5 output: " << configuration.outputPath << '\n'
                  << "Absorbed particles: "
                  << solver.absorbedParticleCount << '\n';
        return EXIT_SUCCESS;
    } catch (const H5::Exception& error) {
        std::cerr << "HDF5 error: " << error.getDetailMsg() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
    }
    return EXIT_FAILURE;
}
