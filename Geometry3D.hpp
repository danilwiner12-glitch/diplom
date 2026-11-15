#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct Material {
    std::string name;
    double relativePermittivity;
    double relativePermeability;
    double conductivity;

    Material(std::string name_ = "vacuum",
             double relativePermittivity_ = 1.0,
             double relativePermeability_ = 1.0,
             double conductivity_ = 0.0)
        : name(std::move(name_)),
          relativePermittivity(relativePermittivity_),
          relativePermeability(relativePermeability_),
          conductivity(conductivity_)
    {
        if (relativePermittivity <= 0.0 || relativePermeability <= 0.0) {
            throw std::runtime_error(
                "Relative permittivity and permeability must be positive");
        }
        if (conductivity < 0.0) {
            throw std::runtime_error("Conductivity cannot be negative");
        }
    }

    double refractiveIndex() const noexcept {
        return std::sqrt(relativePermittivity * relativePermeability);
    }
};

struct AxisAlignedBox {
    double xMin;
    double yMin;
    double zMin;
    double xMax;
    double yMax;
    double zMax;
    Material material;

    bool contains(double x, double y, double z) const noexcept {
        return x >= xMin && x <= xMax &&
               y >= yMin && y <= yMax &&
               z >= zMin && z <= zMax;
    }
};

class Geometry3D {
public:
    explicit Geometry3D(Material background = Material())
        : background_(std::move(background))
    {
    }

    const Material& background() const noexcept {
        return background_;
    }

    void setBackground(Material material) {
        background_ = std::move(material);
    }

    void addBox(double xMin, double yMin, double zMin,
                double xMax, double yMax, double zMax,
                Material material) {
        if (xMin >= xMax || yMin >= yMax || zMin >= zMax) {
            throw std::runtime_error("Geometry box must have positive volume");
        }
        boxes_.push_back(AxisAlignedBox{xMin, yMin, zMin,
                                        xMax, yMax, zMax,
                                        std::move(material)});
    }

    const Material& materialAt(double x, double y, double z) const noexcept {
        for (auto box = boxes_.rbegin(); box != boxes_.rend(); ++box) {
            if (box->contains(x, y, z)) {
                return box->material;
            }
        }
        return background_;
    }

    const std::vector<AxisAlignedBox>& boxes() const noexcept {
        return boxes_;
    }

private:
    Material background_;
    std::vector<AxisAlignedBox> boxes_;
};
