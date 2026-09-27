#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>

class YeeGrid3D {
public:
    std::size_t nx;
    std::size_t ny;
    std::size_t nz;

    double lx;
    double ly;
    double lz;

    double dx;
    double dy;
    double dz;

    double invDx;
    double invDy;
    double invDz;

    YeeGrid3D(std::size_t nx_, std::size_t ny_, std::size_t nz_,
              double lx_, double ly_, double lz_)
        : nx(nx_),
          ny(ny_),
          nz(nz_),
          lx(lx_),
          ly(ly_),
          lz(lz_),
          dx(0.0),
          dy(0.0),
          dz(0.0),
          invDx(0.0),
          invDy(0.0),
          invDz(0.0)
    {
        if (nx < 2 || ny < 2 || nz < 2) {
            throw std::runtime_error("Yee grid must have at least 2x2x2 cells");
        }
        if (lx <= 0.0 || ly <= 0.0 || lz <= 0.0) {
            throw std::runtime_error("Yee grid physical sizes must be positive");
        }

        dx = lx / static_cast<double>(nx);
        dy = ly / static_cast<double>(ny);
        dz = lz / static_cast<double>(nz);

        invDx = 1.0 / dx;
        invDy = 1.0 / dy;
        invDz = 1.0 / dz;
    }

    inline constexpr std::size_t index(std::size_t i,
                                       std::size_t j,
                                       std::size_t k,
                                       std::size_t sx,
                                       std::size_t sy) const noexcept
    {
        return i + sx * (j + sy * k);
    }

    inline constexpr std::size_t indexEx(std::size_t i, std::size_t j, std::size_t k) const noexcept {
        return index(i, j, k, nx, ny + 1);
    }

    inline constexpr std::size_t indexEy(std::size_t i, std::size_t j, std::size_t k) const noexcept {
        return index(i, j, k, nx + 1, ny);
    }

    inline constexpr std::size_t indexEz(std::size_t i, std::size_t j, std::size_t k) const noexcept {
        return index(i, j, k, nx + 1, ny + 1);
    }

    inline constexpr std::size_t indexBx(std::size_t i, std::size_t j, std::size_t k) const noexcept {
        return index(i, j, k, nx + 1, ny);
    }

    inline constexpr std::size_t indexBy(std::size_t i, std::size_t j, std::size_t k) const noexcept {
        return index(i, j, k, nx, ny + 1);
    }

    inline constexpr std::size_t indexBz(std::size_t i, std::size_t j, std::size_t k) const noexcept {
        return index(i, j, k, nx, ny);
    }

    inline constexpr std::size_t sizeEx() const noexcept {
        return nx * (ny + 1) * (nz + 1);
    }

    inline constexpr std::size_t sizeEy() const noexcept {
        return (nx + 1) * ny * (nz + 1);
    }

    inline constexpr std::size_t sizeEz() const noexcept {
        return (nx + 1) * (ny + 1) * nz;
    }

    inline constexpr std::size_t sizeBx() const noexcept {
        return (nx + 1) * ny * nz;
    }

    inline constexpr std::size_t sizeBy() const noexcept {
        return nx * (ny + 1) * nz;
    }

    inline constexpr std::size_t sizeBz() const noexcept {
        return nx * ny * (nz + 1);
    }

    inline constexpr double xEx(std::size_t i) const noexcept { return (static_cast<double>(i) + 0.5) * dx; }
    inline constexpr double yEx(std::size_t j) const noexcept { return static_cast<double>(j) * dy; }
    inline constexpr double zEx(std::size_t k) const noexcept { return static_cast<double>(k) * dz; }

    inline constexpr double xEy(std::size_t i) const noexcept { return static_cast<double>(i) * dx; }
    inline constexpr double yEy(std::size_t j) const noexcept { return (static_cast<double>(j) + 0.5) * dy; }
    inline constexpr double zEy(std::size_t k) const noexcept { return static_cast<double>(k) * dz; }

    inline constexpr double xEz(std::size_t i) const noexcept { return static_cast<double>(i) * dx; }
    inline constexpr double yEz(std::size_t j) const noexcept { return static_cast<double>(j) * dy; }
    inline constexpr double zEz(std::size_t k) const noexcept { return (static_cast<double>(k) + 0.5) * dz; }

    inline constexpr double xBx(std::size_t i) const noexcept { return static_cast<double>(i) * dx; }
    inline constexpr double yBx(std::size_t j) const noexcept { return (static_cast<double>(j) + 0.5) * dy; }
    inline constexpr double zBx(std::size_t k) const noexcept { return (static_cast<double>(k) + 0.5) * dz; }

    inline constexpr double xBy(std::size_t i) const noexcept { return (static_cast<double>(i) + 0.5) * dx; }
    inline constexpr double yBy(std::size_t j) const noexcept { return static_cast<double>(j) * dy; }
    inline constexpr double zBy(std::size_t k) const noexcept { return (static_cast<double>(k) + 0.5) * dz; }

    inline constexpr double xBz(std::size_t i) const noexcept { return (static_cast<double>(i) + 0.5) * dx; }
    inline constexpr double yBz(std::size_t j) const noexcept { return (static_cast<double>(j) + 0.5) * dy; }
    inline constexpr double zBz(std::size_t k) const noexcept { return static_cast<double>(k) * dz; }
};

struct YeeGrid3DInitialization {
    YeeGrid3D grid;
    double dt;
    double cflLimitDt;
};

inline YeeGrid3DInitialization initializeYeeGrid3D(double lx,
                                                   double ly,
                                                   double lz,
                                                   double minWavelength,
                                                   double c0 = 299792458.0,
                                                   std::size_t pointsPerWavelength = 20,
                                                   double cflSafety = 0.98)
{
    if (lx <= 0.0 || ly <= 0.0 || lz <= 0.0) {
        throw std::runtime_error("Physical grid sizes must be positive");
    }
    if (minWavelength <= 0.0) {
        throw std::runtime_error("Minimum wavelength must be positive");
    }
    if (c0 <= 0.0) {
        throw std::runtime_error("Wave speed must be positive");
    }
    if (pointsPerWavelength < 2) {
        throw std::runtime_error("Use at least 2 points per wavelength");
    }
    if (cflSafety <= 0.0 || cflSafety >= 1.0) {
        throw std::runtime_error("CFL safety factor must be in the interval (0, 1)");
    }

    const double targetStep = minWavelength / static_cast<double>(pointsPerWavelength);
    const auto nx = static_cast<std::size_t>(std::ceil(lx / targetStep));
    const auto ny = static_cast<std::size_t>(std::ceil(ly / targetStep));
    const auto nz = static_cast<std::size_t>(std::ceil(lz / targetStep));

    YeeGrid3D grid(nx < 2 ? 2 : nx,
                   ny < 2 ? 2 : ny,
                   nz < 2 ? 2 : nz,
                   lx,
                   ly,
                   lz);

    const double invDx2 = 1.0 / (grid.dx * grid.dx);
    const double invDy2 = 1.0 / (grid.dy * grid.dy);
    const double invDz2 = 1.0 / (grid.dz * grid.dz);
    const double cflLimitDt = 1.0 / (c0 * std::sqrt(invDx2 + invDy2 + invDz2));
    const double dt = cflSafety * cflLimitDt;

    return YeeGrid3DInitialization{grid, dt, cflLimitDt};
}