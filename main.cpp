#include <cstddef>
#include <iostream>
#include "YeeGrid3D.hpp"
#include "Field3D.hpp"
#include "Solver3D.hpp"

int main() {
    const double lx = 1.0;
    const double ly = 0.8;
    const double lz = 0.6;
    const double minWavelength = 0.05;
    const std::size_t pointsPerWavelength = 20;
    const double cflSafety = 0.98;

    const auto init = initializeYeeGrid3D(lx,
                                          ly,
                                          lz,
                                          minWavelength,
                                          299792458.0,
                                          pointsPerWavelength,
                                          cflSafety);

    const YeeGrid3D& grid = init.grid;
    const double dt = init.dt;

    Field3D field(grid);

    std::cout << "3D Yee grid created\n";

    std::cout << "nx = " << grid.nx << "\n";
    std::cout << "ny = " << grid.ny << "\n";
    std::cout << "nz = " << grid.nz << "\n";

    std::cout << "dx = " << grid.dx << "\n";
    std::cout << "dy = " << grid.dy << "\n";
    std::cout << "dz = " << grid.dz << "\n";
    std::cout << "CFL dt limit = " << init.cflLimitDt << "\n";
    std::cout << "dt = " << dt << "\n";

    std::cout << "Ex size = " << field.Ex.size() << "\n";
    std::cout << "Ey size = " << field.Ey.size() << "\n";
    std::cout << "Ez size = " << field.Ez.size() << "\n";
    std::cout << "Bx size = " << field.Bx.size() << "\n";
    std::cout << "By size = " << field.By.size() << "\n";
    std::cout << "Bz size = " << field.Bz.size() << "\n";

    const std::size_t i = 10;
    const std::size_t j = 20;
    const std::size_t k = 30;

    const std::size_t idEz = grid.indexEz(i, j, k);
    field.Ez[idEz] = 1.0;

    const std::size_t idBz = grid.indexBz(i, j, k);
    field.Bz[idBz] = 0.5;

    std::cout << "Test Ez(" << i << ", " << j << ", " << k << ") = "
              << field.Ez[idEz] << "\n";
    std::cout << "Test Bz(" << i << ", " << j << ", " << k << ") = "
              << field.Bz[idBz] << "\n";

    std::cout << "Ez coordinate: x = " << grid.xEz(i)
              << ", y = " << grid.yEz(j)
              << ", z = " << grid.zEz(k) << "\n";

    std::cout << "Bz coordinate: x = " << grid.xBz(i)
              << ", y = " << grid.yBz(j)
              << ", z = " << grid.zBz(k) << "\n";

    Solver3D solver(grid, field, dt);
    solver.addParticle(0.5 * lx, 0.5 * ly, 0.5 * lz,
                       0.0, 0.0, 0.0,
                       -1.602176634e-19, 9.1093837015e-31);

    solver.step();

    std::cout << "Particles = " << solver.particles.size() << "\n";

    return 0;
}