#pragma once

#include <algorithm>
#include <vector>
#include "YeeGrid3D.hpp"

class Field3D {
public:
    std::vector<double> Ex;
    std::vector<double> Ey;
    std::vector<double> Ez;

    std::vector<double> Bx;
    std::vector<double> By;
    std::vector<double> Bz;

    std::vector<double> Jx;
    std::vector<double> Jy;
    std::vector<double> Jz;
    std::vector<double> rho;

    explicit Field3D(const YeeGrid3D& grid)
        : Ex(grid.sizeEx(), 0.0),
          Ey(grid.sizeEy(), 0.0),
          Ez(grid.sizeEz(), 0.0),
          Bx(grid.sizeBx(), 0.0),
          By(grid.sizeBy(), 0.0),
          Bz(grid.sizeBz(), 0.0),
          Jx(grid.sizeEx(), 0.0),
          Jy(grid.sizeEy(), 0.0),
          Jz(grid.sizeEz(), 0.0),
          rho((grid.nx + 1) * (grid.ny + 1) * (grid.nz + 1), 0.0)
    {
    }

    void clear() {
        std::fill(Ex.begin(), Ex.end(), 0.0);
        std::fill(Ey.begin(), Ey.end(), 0.0);
        std::fill(Ez.begin(), Ez.end(), 0.0);
        std::fill(Bx.begin(), Bx.end(), 0.0);
        std::fill(By.begin(), By.end(), 0.0);
        std::fill(Bz.begin(), Bz.end(), 0.0);
        clearSources();
    }

    void clearSources() {
        std::fill(Jx.begin(), Jx.end(), 0.0);
        std::fill(Jy.begin(), Jy.end(), 0.0);
        std::fill(Jz.begin(), Jz.end(), 0.0);
        std::fill(rho.begin(), rho.end(), 0.0);
    }
};