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

    explicit Field3D(const YeeGrid3D& grid)
        : Ex(grid.sizeEx(), 0.0),
          Ey(grid.sizeEy(), 0.0),
          Ez(grid.sizeEz(), 0.0),
          Bx(grid.sizeBx(), 0.0),
          By(grid.sizeBy(), 0.0),
          Bz(grid.sizeBz(), 0.0)
    {
    }

    void clear() {
        std::fill(Ex.begin(), Ex.end(), 0.0);
        std::fill(Ey.begin(), Ey.end(), 0.0);
        std::fill(Ez.begin(), Ez.end(), 0.0);
        std::fill(Bx.begin(), Bx.end(), 0.0);
        std::fill(By.begin(), By.end(), 0.0);
        std::fill(Bz.begin(), Bz.end(), 0.0);
    }
};
