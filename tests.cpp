#include <algorithm>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#include "Field3D.hpp"
#include "Geometry3D.hpp"
#include "Solver3D.hpp"
#include "YeeGrid3D.hpp"

namespace {
constexpr double c0 = 299792458.0;

double continuityRelativeResidual(const YeeGrid3D& grid,
                                  const Field3D& field,
                                  const std::vector<double>& oldRho,
                                  double dt) {
    double maximumResidual = 0.0;
    double maximumScale = 0.0;
    for (std::size_t k = 0; k <= grid.nz; ++k) {
        for (std::size_t j = 0; j <= grid.ny; ++j) {
            for (std::size_t i = 0; i <= grid.nx; ++i) {
                const std::size_t id = grid.index(i, j, k,
                                                  grid.nx + 1,
                                                  grid.ny + 1);
                const double chargeChange =
                    (field.rho[id] - oldRho[id]) / dt;
                const double jxRight = i < grid.nx
                    ? field.Jx[grid.indexEx(i, j, k)] : 0.0;
                const double jxLeft = i > 0
                    ? field.Jx[grid.indexEx(i - 1, j, k)] : 0.0;
                const double jyRight = j < grid.ny
                    ? field.Jy[grid.indexEy(i, j, k)] : 0.0;
                const double jyLeft = j > 0
                    ? field.Jy[grid.indexEy(i, j - 1, k)] : 0.0;
                const double jzRight = k < grid.nz
                    ? field.Jz[grid.indexEz(i, j, k)] : 0.0;
                const double jzLeft = k > 0
                    ? field.Jz[grid.indexEz(i, j, k - 1)] : 0.0;
                const double divergence =
                    (jxRight - jxLeft) * grid.invDx +
                    (jyRight - jyLeft) * grid.invDy +
                    (jzRight - jzLeft) * grid.invDz;
                maximumResidual = std::max(
                    maximumResidual,
                    std::abs(chargeChange + divergence));
                maximumScale = std::max(
                    maximumScale,
                    std::abs(chargeChange) + std::abs(divergence));
            }
        }
    }
    return maximumScale == 0.0 ? 0.0 : maximumResidual / maximumScale;
}

double totalCharge(const YeeGrid3D& grid, const Field3D& field) {
    double charge = 0.0;
    for (const double density : field.rho) {
        charge += density * grid.dx * grid.dy * grid.dz;
    }
    return charge;
}

void testGeometry() {
    Geometry3D geometry(Material("background", 2.25, 1.0, 0.0));
    geometry.addBox(0.2, 0.2, 0.2, 0.8, 0.8, 0.8,
                    Material("object", 4.0, 1.0, 0.1));
    assert(geometry.materialAt(0.1, 0.1, 0.1).name == "background");
    assert(geometry.materialAt(0.5, 0.5, 0.5).name == "object");
    assert(std::abs(geometry.background().refractiveIndex() - 1.5) <
           1.0e-15);
}

void testInteriorChargeConservation() {
    const YeeGrid3D grid(4, 4, 4, 4.0, 4.0, 4.0);
    Field3D field(grid);
    const double dt = 1.0e-9;
    Solver3D solver(grid, field, dt);
    const double vx = 0.60 * c0;
    const double vy = 0.20 * c0;
    const double vz = -0.10 * c0;
    const double speedSquared = vx * vx + vy * vy + vz * vz;
    const double gamma =
        1.0 / std::sqrt(1.0 - speedSquared / (c0 * c0));
    solver.addParticle(0.90, 1.90, 2.10,
                       gamma * vx, gamma * vy, gamma * vz,
                       1.0, 1.0);
    solver.initializeSources();
    const std::vector<double> oldRho = field.rho;
    solver.step();
    assert(continuityRelativeResidual(grid, field, oldRho, dt) < 1.0e-14);
    assert(std::abs(totalCharge(grid, field) - 1.0) < 1.0e-14);
}

void testAbsorbingParticleBoundary() {
    const YeeGrid3D grid(4, 4, 4, 4.0, 4.0, 4.0);
    Field3D field(grid);
    const double dt = 1.0e-9;
    Solver3D solver(grid, field, dt);
    const double vx = -0.50 * c0;
    const double gamma = 1.0 / std::sqrt(1.0 - 0.25);
    solver.addParticle(0.01, 2.0, 2.0,
                       gamma * vx, 0.0, 0.0, 1.0, 1.0);
    solver.initializeSources();
    const std::vector<double> oldRho = field.rho;
    solver.step();
    assert(solver.particles.empty());
    assert(solver.absorbedParticleCount == 1);
    assert(std::abs(solver.absorbedCharge - 1.0) < 1.0e-15);
    assert(std::abs(totalCharge(grid, field) - 1.0) < 1.0e-14);
    assert(continuityRelativeResidual(grid, field, oldRho, dt) < 1.0e-14);
}

void testPerfectElectricConductorBoundary() {
    const YeeGrid3D grid(3, 3, 3, 3.0, 3.0, 3.0);
    Field3D field(grid);
    std::fill(field.Ex.begin(), field.Ex.end(), 1.0);
    std::fill(field.Ey.begin(), field.Ey.end(), 1.0);
    std::fill(field.Ez.begin(), field.Ez.end(), 1.0);
    std::fill(field.Bx.begin(), field.Bx.end(), 1.0);
    std::fill(field.By.begin(), field.By.end(), 1.0);
    std::fill(field.Bz.begin(), field.Bz.end(), 1.0);
    Solver3D solver(grid, field, 1.0e-12);
    solver.step();

    for (std::size_t k = 0; k <= grid.nz; ++k) {
        for (std::size_t j = 0; j < grid.ny; ++j) {
            assert(field.Ey[grid.indexEy(0, j, k)] == 0.0);
            assert(field.Ey[grid.indexEy(grid.nx, j, k)] == 0.0);
        }
    }
    for (std::size_t k = 0; k < grid.nz; ++k) {
        for (std::size_t j = 0; j <= grid.ny; ++j) {
            assert(field.Ez[grid.indexEz(0, j, k)] == 0.0);
            assert(field.Ez[grid.indexEz(grid.nx, j, k)] == 0.0);
        }
    }
    for (std::size_t k = 0; k < grid.nz; ++k) {
        for (std::size_t j = 0; j < grid.ny; ++j) {
            assert(field.Bx[grid.indexBx(0, j, k)] == 0.0);
            assert(field.Bx[grid.indexBx(grid.nx, j, k)] == 0.0);
        }
    }
    for (std::size_t k = 0; k < grid.nz; ++k) {
        for (std::size_t i = 0; i < grid.nx; ++i) {
            assert(field.By[grid.indexBy(i, 0, k)] == 0.0);
            assert(field.By[grid.indexBy(i, grid.ny, k)] == 0.0);
        }
    }
    for (std::size_t j = 0; j < grid.ny; ++j) {
        for (std::size_t i = 0; i < grid.nx; ++i) {
            assert(field.Bz[grid.indexBz(i, j, 0)] == 0.0);
            assert(field.Bz[grid.indexBz(i, j, grid.nz)] == 0.0);
        }
    }
}

void testHomogeneousDielectricFieldUpdate() {
    const YeeGrid3D grid(3, 3, 3, 3.0, 3.0, 3.0);
    Field3D vacuumField(grid);
    Field3D dielectricField(grid);
    for (std::size_t k = 0; k <= grid.nz; ++k) {
        for (std::size_t j = 0; j < grid.ny; ++j) {
            for (std::size_t i = 0; i < grid.nx; ++i) {
                const double value = static_cast<double>(j);
                vacuumField.Bz[grid.indexBz(i, j, k)] = value;
                dielectricField.Bz[grid.indexBz(i, j, k)] = value;
            }
        }
    }

    const double dt = 1.0e-12;
    Solver3D vacuumSolver(grid, vacuumField, dt);
    Solver3D dielectricSolver(
        grid, dielectricField, dt,
        Geometry3D(Material("dielectric", 4.0, 1.0, 0.0)));
    vacuumSolver.step();
    dielectricSolver.step();

    const std::size_t id = grid.indexEx(1, 1, 1);
    assert(vacuumField.Ex[id] != 0.0);
    assert(std::abs(dielectricField.Ex[id] / vacuumField.Ex[id] - 0.25) <
           1.0e-14);
}
}

int main() {
    testGeometry();
    testInteriorChargeConservation();
    testAbsorbingParticleBoundary();
    testPerfectElectricConductorBoundary();
    testHomogeneousDielectricFieldUpdate();
    std::cout << "All tests passed\n";
}
