#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include "YeeGrid3D.hpp"
#include "Field3D.hpp"

struct Particle {
    double x;
    double y;
    double z;
    double vx;
    double vy;
    double vz;
    double q;
    double m;
};

class Solver3D {
public:
    const YeeGrid3D& grid;
    Field3D& field;
    double dt;
    std::vector<Particle> particles;

    Solver3D(const YeeGrid3D& grid_, Field3D& field_, double dt_)
        : grid(grid_), field(field_), dt(dt_)
    {
        if (dt <= 0.0) {
            throw std::runtime_error("Time step must be positive");
        }
    }

    void addParticle(const Particle& particle) {
        if (particle.m <= 0.0) {
            throw std::runtime_error("Particle mass must be positive");
        }
        if (particle.x < 0.0 || particle.x > grid.lx ||
            particle.y < 0.0 || particle.y > grid.ly ||
            particle.z < 0.0 || particle.z > grid.lz) {
            throw std::runtime_error("Particle is outside the grid");
        }
        particles.push_back(particle);
    }

    void addParticle(double x, double y, double z,
                     double vx, double vy, double vz,
                     double q, double m) {
        addParticle(Particle{x, y, z, vx, vy, vz, q, m});
    }

    void step() {
        updateMagneticField();
        moveParticles();
        depositParticles();
        updateElectricField();
        applyFieldBoundaryConditions();
    }

private:
    static constexpr double c0 = 299792458.0;
    static constexpr double epsilon0 = 8.8541878128e-12;

    std::size_t flatIndex(std::size_t i, std::size_t j, std::size_t k,
                          std::size_t sx, std::size_t sy) const {
        return i + sx * (j + sy * k);
    }

    double sample(const std::vector<double>& data,
                  std::size_t sx, std::size_t sy, std::size_t sz,
                  double ox, double oy, double oz,
                  double x, double y, double z) const {
        double gx = x * grid.invDx - ox;
        double gy = y * grid.invDy - oy;
        double gz = z * grid.invDz - oz;

        std::size_t i0;
        std::size_t j0;
        std::size_t k0;
        double tx;
        double ty;
        double tz;

        getCell(gx, sx, i0, tx);
        getCell(gy, sy, j0, ty);
        getCell(gz, sz, k0, tz);

        double value = 0.0;
        for (std::size_t dk = 0; dk < 2; ++dk) {
            for (std::size_t dj = 0; dj < 2; ++dj) {
                for (std::size_t di = 0; di < 2; ++di) {
                    const double wx = di == 0 ? 1.0 - tx : tx;
                    const double wy = dj == 0 ? 1.0 - ty : ty;
                    const double wz = dk == 0 ? 1.0 - tz : tz;
                    value += wx * wy * wz *
                             data[flatIndex(i0 + di, j0 + dj, k0 + dk, sx, sy)];
                }
            }
        }
        return value;
    }

    void deposit(std::vector<double>& data,
                 std::size_t sx, std::size_t sy, std::size_t sz,
                 double ox, double oy, double oz,
                 double x, double y, double z, double value) {
        double gx = x * grid.invDx - ox;
        double gy = y * grid.invDy - oy;
        double gz = z * grid.invDz - oz;

        std::size_t i0;
        std::size_t j0;
        std::size_t k0;
        double tx;
        double ty;
        double tz;

        getCell(gx, sx, i0, tx);
        getCell(gy, sy, j0, ty);
        getCell(gz, sz, k0, tz);

        for (std::size_t dk = 0; dk < 2; ++dk) {
            for (std::size_t dj = 0; dj < 2; ++dj) {
                for (std::size_t di = 0; di < 2; ++di) {
                    const double wx = di == 0 ? 1.0 - tx : tx;
                    const double wy = dj == 0 ? 1.0 - ty : ty;
                    const double wz = dk == 0 ? 1.0 - tz : tz;
                    data[flatIndex(i0 + di, j0 + dj, k0 + dk, sx, sy)] +=
                        value * wx * wy * wz;
                }
            }
        }
    }

    void getCell(double g, std::size_t size, std::size_t& i0, double& t) const {
        if (g <= 0.0) {
            i0 = 0;
            t = 0.0;
            return;
        }
        if (g >= static_cast<double>(size - 1)) {
            i0 = size - 2;
            t = 1.0;
            return;
        }
        i0 = static_cast<std::size_t>(std::floor(g));
        t = g - static_cast<double>(i0);
    }

    void updateMagneticField() {
        for (std::size_t k = 0; k < grid.nz; ++k) {
            for (std::size_t j = 0; j < grid.ny; ++j) {
                for (std::size_t i = 0; i <= grid.nx; ++i) {
                    const double dEzDy = (field.Ez[grid.indexEz(i, j + 1, k)] -
                                          field.Ez[grid.indexEz(i, j, k)]) * grid.invDy;
                    const double dEyDz = (field.Ey[grid.indexEy(i, j, k + 1)] -
                                          field.Ey[grid.indexEy(i, j, k)]) * grid.invDz;
                    field.Bx[grid.indexBx(i, j, k)] -= dt * (dEzDy - dEyDz);
                }
            }
        }

        for (std::size_t k = 0; k < grid.nz; ++k) {
            for (std::size_t j = 0; j <= grid.ny; ++j) {
                for (std::size_t i = 0; i < grid.nx; ++i) {
                    const double dExDz = (field.Ex[grid.indexEx(i, j, k + 1)] -
                                          field.Ex[grid.indexEx(i, j, k)]) * grid.invDz;
                    const double dEzDx = (field.Ez[grid.indexEz(i + 1, j, k)] -
                                          field.Ez[grid.indexEz(i, j, k)]) * grid.invDx;
                    field.By[grid.indexBy(i, j, k)] -= dt * (dExDz - dEzDx);
                }
            }
        }

        for (std::size_t k = 0; k <= grid.nz; ++k) {
            for (std::size_t j = 0; j < grid.ny; ++j) {
                for (std::size_t i = 0; i < grid.nx; ++i) {
                    const double dEyDx = (field.Ey[grid.indexEy(i + 1, j, k)] -
                                          field.Ey[grid.indexEy(i, j, k)]) * grid.invDx;
                    const double dExDy = (field.Ex[grid.indexEx(i, j + 1, k)] -
                                          field.Ex[grid.indexEx(i, j, k)]) * grid.invDy;
                    field.Bz[grid.indexBz(i, j, k)] -= dt * (dEyDx - dExDy);
                }
            }
        }
    }

    void updateElectricField() {
        const double c2dt = c0 * c0 * dt;
        const double currentDt = dt / epsilon0;

        for (std::size_t k = 1; k < grid.nz; ++k) {
            for (std::size_t j = 1; j < grid.ny; ++j) {
                for (std::size_t i = 0; i < grid.nx; ++i) {
                    const std::size_t id = grid.indexEx(i, j, k);
                    const double dBzDy = (field.Bz[grid.indexBz(i, j, k)] -
                                          field.Bz[grid.indexBz(i, j - 1, k)]) * grid.invDy;
                    const double dByDz = (field.By[grid.indexBy(i, j, k)] -
                                          field.By[grid.indexBy(i, j, k - 1)]) * grid.invDz;
                    field.Ex[id] += c2dt * (dBzDy - dByDz) - currentDt * field.Jx[id];
                }
            }
        }

        for (std::size_t k = 1; k < grid.nz; ++k) {
            for (std::size_t j = 0; j < grid.ny; ++j) {
                for (std::size_t i = 1; i < grid.nx; ++i) {
                    const std::size_t id = grid.indexEy(i, j, k);
                    const double dBxDz = (field.Bx[grid.indexBx(i, j, k)] -
                                          field.Bx[grid.indexBx(i, j, k - 1)]) * grid.invDz;
                    const double dBzDx = (field.Bz[grid.indexBz(i, j, k)] -
                                          field.Bz[grid.indexBz(i - 1, j, k)]) * grid.invDx;
                    field.Ey[id] += c2dt * (dBxDz - dBzDx) - currentDt * field.Jy[id];
                }
            }
        }

        for (std::size_t k = 0; k < grid.nz; ++k) {
            for (std::size_t j = 1; j < grid.ny; ++j) {
                for (std::size_t i = 1; i < grid.nx; ++i) {
                    const std::size_t id = grid.indexEz(i, j, k);
                    const double dByDx = (field.By[grid.indexBy(i, j, k)] -
                                          field.By[grid.indexBy(i - 1, j, k)]) * grid.invDx;
                    const double dBxDy = (field.Bx[grid.indexBx(i, j, k)] -
                                          field.Bx[grid.indexBx(i, j - 1, k)]) * grid.invDy;
                    field.Ez[id] += c2dt * (dByDx - dBxDy) - currentDt * field.Jz[id];
                }
            }
        }
    }

    void moveParticles() {
        for (Particle& p : particles) {
            const double ex = sample(field.Ex, grid.nx, grid.ny + 1, grid.nz + 1,
                                     0.5, 0.0, 0.0, p.x, p.y, p.z);
            const double ey = sample(field.Ey, grid.nx + 1, grid.ny, grid.nz + 1,
                                     0.0, 0.5, 0.0, p.x, p.y, p.z);
            const double ez = sample(field.Ez, grid.nx + 1, grid.ny + 1, grid.nz,
                                     0.0, 0.0, 0.5, p.x, p.y, p.z);

            const double bx = sample(field.Bx, grid.nx + 1, grid.ny, grid.nz,
                                     0.0, 0.5, 0.5, p.x, p.y, p.z);
            const double by = sample(field.By, grid.nx, grid.ny + 1, grid.nz,
                                     0.5, 0.0, 0.5, p.x, p.y, p.z);
            const double bz = sample(field.Bz, grid.nx, grid.ny, grid.nz + 1,
                                     0.5, 0.5, 0.0, p.x, p.y, p.z);

            const double a = p.q * dt / (2.0 * p.m);

            const double vmx = p.vx + a * ex;
            const double vmy = p.vy + a * ey;
            const double vmz = p.vz + a * ez;

            const double tx = a * bx;
            const double ty = a * by;
            const double tz = a * bz;
            const double t2 = tx * tx + ty * ty + tz * tz;

            const double sx = 2.0 * tx / (1.0 + t2);
            const double sy = 2.0 * ty / (1.0 + t2);
            const double sz = 2.0 * tz / (1.0 + t2);

            const double vpx = vmx + (vmy * tz - vmz * ty);
            const double vpy = vmy + (vmz * tx - vmx * tz);
            const double vpz = vmz + (vmx * ty - vmy * tx);

            const double vplusx = vmx + (vpy * sz - vpz * sy);
            const double vplusy = vmy + (vpz * sx - vpx * sz);
            const double vplusz = vmz + (vpx * sy - vpy * sx);

            p.vx = vplusx + a * ex;
            p.vy = vplusy + a * ey;
            p.vz = vplusz + a * ez;

            p.x += p.vx * dt;
            p.y += p.vy * dt;
            p.z += p.vz * dt;

            reflectParticle(p);
        }
    }

    void reflectParticle(Particle& p) {
        if (p.x < 0.0) {
            p.x = -p.x;
            p.vx = -p.vx;
        } else if (p.x > grid.lx) {
            p.x = 2.0 * grid.lx - p.x;
            p.vx = -p.vx;
        }

        if (p.y < 0.0) {
            p.y = -p.y;
            p.vy = -p.vy;
        } else if (p.y > grid.ly) {
            p.y = 2.0 * grid.ly - p.y;
            p.vy = -p.vy;
        }

        if (p.z < 0.0) {
            p.z = -p.z;
            p.vz = -p.vz;
        } else if (p.z > grid.lz) {
            p.z = 2.0 * grid.lz - p.z;
            p.vz = -p.vz;
        }
    }

    void depositParticles() {
        field.clearSources();
        const double invVolume = grid.invDx * grid.invDy * grid.invDz;

        for (const Particle& p : particles) {
            deposit(field.rho, grid.nx + 1, grid.ny + 1, grid.nz + 1,
                    0.0, 0.0, 0.0, p.x, p.y, p.z, p.q * invVolume);

            deposit(field.Jx, grid.nx, grid.ny + 1, grid.nz + 1,
                    0.5, 0.0, 0.0, p.x, p.y, p.z, p.q * p.vx * invVolume);
            deposit(field.Jy, grid.nx + 1, grid.ny, grid.nz + 1,
                    0.0, 0.5, 0.0, p.x, p.y, p.z, p.q * p.vy * invVolume);
            deposit(field.Jz, grid.nx + 1, grid.ny + 1, grid.nz,
                    0.0, 0.0, 0.5, p.x, p.y, p.z, p.q * p.vz * invVolume);
        }
    }

    void applyFieldBoundaryConditions() {
        for (std::size_t k = 0; k <= grid.nz; ++k) {
            for (std::size_t j = 0; j < grid.ny; ++j) {
                field.Ey[grid.indexEy(0, j, k)] = 0.0;
                field.Ey[grid.indexEy(grid.nx, j, k)] = 0.0;
            }
        }
        for (std::size_t k = 0; k < grid.nz; ++k) {
            for (std::size_t j = 0; j <= grid.ny; ++j) {
                field.Ez[grid.indexEz(0, j, k)] = 0.0;
                field.Ez[grid.indexEz(grid.nx, j, k)] = 0.0;
            }
        }

        for (std::size_t k = 0; k <= grid.nz; ++k) {
            for (std::size_t i = 0; i < grid.nx; ++i) {
                field.Ex[grid.indexEx(i, 0, k)] = 0.0;
                field.Ex[grid.indexEx(i, grid.ny, k)] = 0.0;
            }
        }
        for (std::size_t k = 0; k < grid.nz; ++k) {
            for (std::size_t i = 0; i <= grid.nx; ++i) {
                field.Ez[grid.indexEz(i, 0, k)] = 0.0;
                field.Ez[grid.indexEz(i, grid.ny, k)] = 0.0;
            }
        }

        for (std::size_t j = 0; j <= grid.ny; ++j) {
            for (std::size_t i = 0; i < grid.nx; ++i) {
                field.Ex[grid.indexEx(i, j, 0)] = 0.0;
                field.Ex[grid.indexEx(i, j, grid.nz)] = 0.0;
            }
        }
        for (std::size_t j = 0; j < grid.ny; ++j) {
            for (std::size_t i = 0; i <= grid.nx; ++i) {
                field.Ey[grid.indexEy(i, j, 0)] = 0.0;
                field.Ey[grid.indexEy(i, j, grid.nz)] = 0.0;
            }
        }

        for (std::size_t k = 0; k < grid.nz; ++k) {
            for (std::size_t j = 0; j < grid.ny; ++j) {
                field.Bx[grid.indexBx(0, j, k)] = 0.0;
                field.Bx[grid.indexBx(grid.nx, j, k)] = 0.0;
            }
        }
        for (std::size_t k = 0; k < grid.nz; ++k) {
            for (std::size_t i = 0; i < grid.nx; ++i) {
                field.By[grid.indexBy(i, 0, k)] = 0.0;
                field.By[grid.indexBy(i, grid.ny, k)] = 0.0;
            }
        }
        for (std::size_t j = 0; j < grid.ny; ++j) {
            for (std::size_t i = 0; i < grid.nx; ++i) {
                field.Bz[grid.indexBz(i, j, 0)] = 0.0;
                field.Bz[grid.indexBz(i, j, grid.nz)] = 0.0;
            }
        }
    }
};
