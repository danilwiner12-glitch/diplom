#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>
#include "YeeGrid3D.hpp"
#include "Field3D.hpp"
#include "Geometry3D.hpp"

struct Particle {
    double x;
    double y;
    double z;
    double px;
    double py;
    double pz;
    double q;
    double m;
};

enum class ParticleBoundaryCondition {
    Absorb,
    Reflect
};

class Solver3D {
public:
    const YeeGrid3D& grid;
    Field3D& field;
    Geometry3D geometry;
    double dt;
    std::vector<Particle> particles;
    ParticleBoundaryCondition particleBoundaryCondition;
    double time = 0.0;
    std::size_t stepNumber = 0;
    std::size_t absorbedParticleCount = 0;
    double absorbedCharge = 0.0;

    Solver3D(const YeeGrid3D& grid_, Field3D& field_, double dt_)
        : Solver3D(grid_, field_, dt_, Geometry3D(),
                   ParticleBoundaryCondition::Absorb)
    {
    }

    Solver3D(const YeeGrid3D& grid_, Field3D& field_, double dt_,
             Geometry3D geometry_,
             ParticleBoundaryCondition particleBoundaryCondition_ =
                 ParticleBoundaryCondition::Absorb)
        : grid(grid_),
          field(field_),
          geometry(std::move(geometry_)),
          dt(dt_),
          particleBoundaryCondition(particleBoundaryCondition_)
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
                     double px, double py, double pz,
                     double q, double m) {
        addParticle(Particle{x, y, z, px, py, pz, q, m});
    }

    void step() {
        updateMagneticField();
        applyMagneticBoundaryConditions();
        const std::vector<Trajectory> trajectories = moveParticles();
        depositParticles(trajectories);
        updateElectricField();
        applyElectricBoundaryConditions();
        time += dt;
        ++stepNumber;
    }

    void initializeSources() {
        depositParticles({});
    }

private:
    static constexpr double c0 = 299792458.0;
    static constexpr double epsilon0 = 8.8541878128e-12;
    static constexpr double mu0 = 1.25663706212e-6;

    struct Position {
        double x;
        double y;
        double z;
    };

    struct ShapeWeight1D {
        std::size_t index;
        double oldValue;
        double delta;
    };

    struct Trajectory {
        Position oldPosition;
        Position newPosition;
        double charge;
    };

    struct SurfaceCharge {
        Position position;
        double charge;
    };

    std::vector<SurfaceCharge> surfaceCharges_;

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
        for (std::size_t k = 1; k < grid.nz; ++k) {
            for (std::size_t j = 1; j < grid.ny; ++j) {
                for (std::size_t i = 0; i < grid.nx; ++i) {
                    const std::size_t id = grid.indexEx(i, j, k);
                    const double hzUpper =
                        field.Bz[grid.indexBz(i, j, k)] *
                        inversePermeabilityAt(grid.xBz(i), grid.yBz(j),
                                              grid.zBz(k));
                    const double hzLower =
                        field.Bz[grid.indexBz(i, j - 1, k)] *
                        inversePermeabilityAt(grid.xBz(i), grid.yBz(j - 1),
                                              grid.zBz(k));
                    const double hyUpper =
                        field.By[grid.indexBy(i, j, k)] *
                        inversePermeabilityAt(grid.xBy(i), grid.yBy(j),
                                              grid.zBy(k));
                    const double hyLower =
                        field.By[grid.indexBy(i, j, k - 1)] *
                        inversePermeabilityAt(grid.xBy(i), grid.yBy(j),
                                              grid.zBy(k - 1));
                    const double curlH = (hzUpper - hzLower) * grid.invDy -
                                         (hyUpper - hyLower) * grid.invDz;
                    advanceElectricComponent(
                        field.Ex[id], field.Jx[id], curlH,
                        geometry.materialAt(grid.xEx(i), grid.yEx(j),
                                            grid.zEx(k)));
                }
            }
        }

        for (std::size_t k = 1; k < grid.nz; ++k) {
            for (std::size_t j = 0; j < grid.ny; ++j) {
                for (std::size_t i = 1; i < grid.nx; ++i) {
                    const std::size_t id = grid.indexEy(i, j, k);
                    const double hxUpper =
                        field.Bx[grid.indexBx(i, j, k)] *
                        inversePermeabilityAt(grid.xBx(i), grid.yBx(j),
                                              grid.zBx(k));
                    const double hxLower =
                        field.Bx[grid.indexBx(i, j, k - 1)] *
                        inversePermeabilityAt(grid.xBx(i), grid.yBx(j),
                                              grid.zBx(k - 1));
                    const double hzUpper =
                        field.Bz[grid.indexBz(i, j, k)] *
                        inversePermeabilityAt(grid.xBz(i), grid.yBz(j),
                                              grid.zBz(k));
                    const double hzLower =
                        field.Bz[grid.indexBz(i - 1, j, k)] *
                        inversePermeabilityAt(grid.xBz(i - 1), grid.yBz(j),
                                              grid.zBz(k));
                    const double curlH = (hxUpper - hxLower) * grid.invDz -
                                         (hzUpper - hzLower) * grid.invDx;
                    advanceElectricComponent(
                        field.Ey[id], field.Jy[id], curlH,
                        geometry.materialAt(grid.xEy(i), grid.yEy(j),
                                            grid.zEy(k)));
                }
            }
        }

        for (std::size_t k = 0; k < grid.nz; ++k) {
            for (std::size_t j = 1; j < grid.ny; ++j) {
                for (std::size_t i = 1; i < grid.nx; ++i) {
                    const std::size_t id = grid.indexEz(i, j, k);
                    const double hyUpper =
                        field.By[grid.indexBy(i, j, k)] *
                        inversePermeabilityAt(grid.xBy(i), grid.yBy(j),
                                              grid.zBy(k));
                    const double hyLower =
                        field.By[grid.indexBy(i - 1, j, k)] *
                        inversePermeabilityAt(grid.xBy(i - 1), grid.yBy(j),
                                              grid.zBy(k));
                    const double hxUpper =
                        field.Bx[grid.indexBx(i, j, k)] *
                        inversePermeabilityAt(grid.xBx(i), grid.yBx(j),
                                              grid.zBx(k));
                    const double hxLower =
                        field.Bx[grid.indexBx(i, j - 1, k)] *
                        inversePermeabilityAt(grid.xBx(i), grid.yBx(j - 1),
                                              grid.zBx(k));
                    const double curlH = (hyUpper - hyLower) * grid.invDx -
                                         (hxUpper - hxLower) * grid.invDy;
                    advanceElectricComponent(
                        field.Ez[id], field.Jz[id], curlH,
                        geometry.materialAt(grid.xEz(i), grid.yEz(j),
                                            grid.zEz(k)));
                }
            }
        }
    }

    double inversePermeabilityAt(double x, double y, double z) const {
        return 1.0 /
               (mu0 * geometry.materialAt(x, y, z).relativePermeability);
    }

    void advanceElectricComponent(double& electricField,
                                  double currentDensity,
                                  double curlH,
                                  const Material& material) const {
        const double permittivity =
            epsilon0 * material.relativePermittivity;
        const double damping =
            0.5 * material.conductivity * dt / permittivity;
        const double oldFieldFactor = (1.0 - damping) / (1.0 + damping);
        const double sourceFactor = dt / (permittivity * (1.0 + damping));
        electricField = oldFieldFactor * electricField +
                        sourceFactor * (curlH - currentDensity);
    }

    std::vector<Trajectory> moveParticles() {
        std::vector<Trajectory> trajectories;
        trajectories.reserve(particles.size());
        std::vector<Particle> survivingParticles;
        survivingParticles.reserve(particles.size());

        for (Particle p : particles) {
            const Position oldPosition{p.x, p.y, p.z};
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

            const double electricImpulse = 0.5 * p.q * dt;

            const double pmx = p.px + electricImpulse * ex;
            const double pmy = p.py + electricImpulse * ey;
            const double pmz = p.pz + electricImpulse * ez;

            const double gammaMinus = momentumGamma(pmx, pmy, pmz, p.m);
            const double magneticRotation =
                p.q * dt / (2.0 * p.m * gammaMinus);
            const double tx = magneticRotation * bx;
            const double ty = magneticRotation * by;
            const double tz = magneticRotation * bz;
            const double t2 = tx * tx + ty * ty + tz * tz;

            const double sx = 2.0 * tx / (1.0 + t2);
            const double sy = 2.0 * ty / (1.0 + t2);
            const double sz = 2.0 * tz / (1.0 + t2);

            const double ppx = pmx + (pmy * tz - pmz * ty);
            const double ppy = pmy + (pmz * tx - pmx * tz);
            const double ppz = pmz + (pmx * ty - pmy * tx);

            const double pplusx = pmx + (ppy * sz - ppz * sy);
            const double pplusy = pmy + (ppz * sx - ppx * sz);
            const double pplusz = pmz + (ppx * sy - ppy * sx);

            p.px = pplusx + electricImpulse * ex;
            p.py = pplusy + electricImpulse * ey;
            p.pz = pplusz + electricImpulse * ez;

            const double gamma = momentumGamma(p.px, p.py, p.pz, p.m);
            const double inverseRelativisticMass = 1.0 / (gamma * p.m);
            p.x += p.px * inverseRelativisticMass * dt;
            p.y += p.py * inverseRelativisticMass * dt;
            p.z += p.pz * inverseRelativisticMass * dt;

            if (isInside(p.x, p.y, p.z)) {
                trajectories.push_back(
                    Trajectory{oldPosition, Position{p.x, p.y, p.z}, p.q});
                survivingParticles.push_back(p);
            } else if (particleBoundaryCondition ==
                       ParticleBoundaryCondition::Reflect) {
                reflectParticle(p);
                trajectories.push_back(
                    Trajectory{oldPosition, Position{p.x, p.y, p.z}, p.q});
                survivingParticles.push_back(p);
            } else {
                const Position boundaryPosition =
                    firstBoundaryIntersection(oldPosition,
                                              Position{p.x, p.y, p.z});
                trajectories.push_back(
                    Trajectory{oldPosition, boundaryPosition, p.q});
                surfaceCharges_.push_back(
                    SurfaceCharge{boundaryPosition, p.q});
                ++absorbedParticleCount;
                absorbedCharge += p.q;
            }
        }

        particles = std::move(survivingParticles);
        return trajectories;
    }

    double momentumGamma(double px, double py, double pz, double mass) const {
        const double momentumMagnitude = std::hypot(px, py, pz);
        return std::hypot(1.0, momentumMagnitude / (mass * c0));
    }

    void reflectParticle(Particle& p) {
        reflectCoordinate(p.x, p.px, grid.lx);
        reflectCoordinate(p.y, p.py, grid.ly);
        reflectCoordinate(p.z, p.pz, grid.lz);
    }

    bool isInside(double x, double y, double z) const noexcept {
        return x >= 0.0 && x <= grid.lx &&
               y >= 0.0 && y <= grid.ly &&
               z >= 0.0 && z <= grid.lz;
    }

    Position firstBoundaryIntersection(const Position& oldPosition,
                                       const Position& newPosition) const {
        const double dx = newPosition.x - oldPosition.x;
        const double dy = newPosition.y - oldPosition.y;
        const double dz = newPosition.z - oldPosition.z;
        double fraction = 1.0;

        const auto includeIntersection = [&fraction](double oldCoordinate,
                                                      double displacement,
                                                      double boundary) {
            if (displacement == 0.0) {
                return;
            }
            const double candidate =
                (boundary - oldCoordinate) / displacement;
            if (candidate >= 0.0 && candidate < fraction) {
                fraction = candidate;
            }
        };

        if (newPosition.x < 0.0) {
            includeIntersection(oldPosition.x, dx, 0.0);
        } else if (newPosition.x > grid.lx) {
            includeIntersection(oldPosition.x, dx, grid.lx);
        }
        if (newPosition.y < 0.0) {
            includeIntersection(oldPosition.y, dy, 0.0);
        } else if (newPosition.y > grid.ly) {
            includeIntersection(oldPosition.y, dy, grid.ly);
        }
        if (newPosition.z < 0.0) {
            includeIntersection(oldPosition.z, dz, 0.0);
        } else if (newPosition.z > grid.lz) {
            includeIntersection(oldPosition.z, dz, grid.lz);
        }

        return Position{
            std::clamp(oldPosition.x + fraction * dx, 0.0, grid.lx),
            std::clamp(oldPosition.y + fraction * dy, 0.0, grid.ly),
            std::clamp(oldPosition.z + fraction * dz, 0.0, grid.lz)};
    }

    void reflectCoordinate(double& coordinate, double& momentum,
                           double length) const {
        if (coordinate >= 0.0 && coordinate <= length) {
            return;
        }

        const double period = 2.0 * length;
        double wrapped = std::fmod(coordinate, period);
        if (wrapped < 0.0) {
            wrapped += period;
        }

        if (wrapped <= length) {
            coordinate = wrapped;
        } else {
            coordinate = period - wrapped;
            momentum = -momentum;
        }
    }

    std::vector<ShapeWeight1D> shapeChange(double oldCoordinate,
                                            double newCoordinate,
                                            double inverseSpacing,
                                            std::size_t nodeCount) const {
        std::vector<ShapeWeight1D> weights;
        weights.reserve(4);

        const auto addWeight = [&weights](std::size_t index,
                                          double oldValue,
                                          double newValue) {
            const auto found = std::find_if(
                weights.begin(), weights.end(),
                [index](const ShapeWeight1D& weight) {
                    return weight.index == index;
                });
            if (found == weights.end()) {
                weights.push_back(ShapeWeight1D{index, oldValue,
                                                newValue - oldValue});
            } else {
                found->oldValue += oldValue;
                found->delta += newValue - oldValue;
            }
        };

        const auto appendCoordinate = [this, inverseSpacing, nodeCount,
                                       &addWeight](double coordinate,
                                                   bool isOld) {
            std::size_t lower;
            double fraction;
            getCell(coordinate * inverseSpacing, nodeCount, lower, fraction);
            const double lowerWeight = 1.0 - fraction;
            if (isOld) {
                addWeight(lower, lowerWeight, 0.0);
                addWeight(lower + 1, fraction, 0.0);
            } else {
                addWeight(lower, 0.0, lowerWeight);
                addWeight(lower + 1, 0.0, fraction);
            }
        };

        appendCoordinate(oldCoordinate, true);
        appendCoordinate(newCoordinate, false);
        std::sort(weights.begin(), weights.end(),
                  [](const ShapeWeight1D& left, const ShapeWeight1D& right) {
                      return left.index < right.index;
                  });
        return weights;
    }

    void depositCurrentX(double charge,
                         const std::vector<ShapeWeight1D>& xWeights,
                         const std::vector<ShapeWeight1D>& yWeights,
                         const std::vector<ShapeWeight1D>& zWeights,
                         double inverseVolume) {
        const double scale = -charge * grid.dx * inverseVolume / dt;
        for (const ShapeWeight1D& y : yWeights) {
            for (const ShapeWeight1D& z : zWeights) {
                const double transverseWeight =
                    y.oldValue * z.oldValue +
                    0.5 * y.delta * z.oldValue +
                    0.5 * y.oldValue * z.delta +
                    (1.0 / 3.0) * y.delta * z.delta;
                double current = 0.0;
                for (std::size_t n = 0; n + 1 < xWeights.size(); ++n) {
                    current += scale * xWeights[n].delta * transverseWeight;
                    for (std::size_t i = xWeights[n].index;
                         i < xWeights[n + 1].index && i < grid.nx; ++i) {
                        field.Jx[grid.indexEx(i, y.index, z.index)] += current;
                    }
                }
            }
        }
    }

    void depositCurrentY(double charge,
                         const std::vector<ShapeWeight1D>& xWeights,
                         const std::vector<ShapeWeight1D>& yWeights,
                         const std::vector<ShapeWeight1D>& zWeights,
                         double inverseVolume) {
        const double scale = -charge * grid.dy * inverseVolume / dt;
        for (const ShapeWeight1D& x : xWeights) {
            for (const ShapeWeight1D& z : zWeights) {
                const double transverseWeight =
                    x.oldValue * z.oldValue +
                    0.5 * x.delta * z.oldValue +
                    0.5 * x.oldValue * z.delta +
                    (1.0 / 3.0) * x.delta * z.delta;
                double current = 0.0;
                for (std::size_t n = 0; n + 1 < yWeights.size(); ++n) {
                    current += scale * yWeights[n].delta * transverseWeight;
                    for (std::size_t j = yWeights[n].index;
                         j < yWeights[n + 1].index && j < grid.ny; ++j) {
                        field.Jy[grid.indexEy(x.index, j, z.index)] += current;
                    }
                }
            }
        }
    }

    void depositCurrentZ(double charge,
                         const std::vector<ShapeWeight1D>& xWeights,
                         const std::vector<ShapeWeight1D>& yWeights,
                         const std::vector<ShapeWeight1D>& zWeights,
                         double inverseVolume) {
        const double scale = -charge * grid.dz * inverseVolume / dt;
        for (const ShapeWeight1D& x : xWeights) {
            for (const ShapeWeight1D& y : yWeights) {
                const double transverseWeight =
                    x.oldValue * y.oldValue +
                    0.5 * x.delta * y.oldValue +
                    0.5 * x.oldValue * y.delta +
                    (1.0 / 3.0) * x.delta * y.delta;
                double current = 0.0;
                for (std::size_t n = 0; n + 1 < zWeights.size(); ++n) {
                    current += scale * zWeights[n].delta * transverseWeight;
                    for (std::size_t k = zWeights[n].index;
                         k < zWeights[n + 1].index && k < grid.nz; ++k) {
                        field.Jz[grid.indexEz(x.index, y.index, k)] += current;
                    }
                }
            }
        }
    }

    void depositParticles(const std::vector<Trajectory>& trajectories) {
        field.clearSources();
        const double invVolume = grid.invDx * grid.invDy * grid.invDz;

        for (const Particle& p : particles) {
            deposit(field.rho, grid.nx + 1, grid.ny + 1, grid.nz + 1,
                    0.0, 0.0, 0.0, p.x, p.y, p.z, p.q * invVolume);
        }
        for (const SurfaceCharge& surfaceCharge : surfaceCharges_) {
            deposit(field.rho, grid.nx + 1, grid.ny + 1, grid.nz + 1,
                    0.0, 0.0, 0.0,
                    surfaceCharge.position.x,
                    surfaceCharge.position.y,
                    surfaceCharge.position.z,
                    surfaceCharge.charge * invVolume);
        }

        for (const Trajectory& trajectory : trajectories) {
            const auto xWeights = shapeChange(trajectory.oldPosition.x,
                                              trajectory.newPosition.x,
                                              grid.invDx,
                                              grid.nx + 1);
            const auto yWeights = shapeChange(trajectory.oldPosition.y,
                                              trajectory.newPosition.y,
                                              grid.invDy,
                                              grid.ny + 1);
            const auto zWeights = shapeChange(trajectory.oldPosition.z,
                                              trajectory.newPosition.z,
                                              grid.invDz,
                                              grid.nz + 1);

            depositCurrentX(trajectory.charge, xWeights, yWeights, zWeights,
                            invVolume);
            depositCurrentY(trajectory.charge, xWeights, yWeights, zWeights,
                            invVolume);
            depositCurrentZ(trajectory.charge, xWeights, yWeights, zWeights,
                            invVolume);
        }
    }

    void applyElectricBoundaryConditions() {
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
    }

    void applyMagneticBoundaryConditions() {
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
