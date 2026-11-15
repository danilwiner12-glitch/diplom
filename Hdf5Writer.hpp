#pragma once

#include <H5Cpp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Field3D.hpp"
#include "Geometry3D.hpp"
#include "Solver3D.hpp"
#include "YeeGrid3D.hpp"

class Hdf5Writer {
public:
    Hdf5Writer(const std::string& path,
               const YeeGrid3D& grid,
               const Geometry3D& geometry)
        : file_(path, H5F_ACC_TRUNC), grid_(grid)
    {
        H5::Exception::dontPrint();
        const H5::Group gridGroup = file_.createGroup("/grid");
        file_.createGroup("/steps");

        writeAttribute(gridGroup, "nx", grid.nx);
        writeAttribute(gridGroup, "ny", grid.ny);
        writeAttribute(gridGroup, "nz", grid.nz);
        writeAttribute(gridGroup, "lx", grid.lx);
        writeAttribute(gridGroup, "ly", grid.ly);
        writeAttribute(gridGroup, "lz", grid.lz);
        writeAttribute(gridGroup, "dx", grid.dx);
        writeAttribute(gridGroup, "dy", grid.dy);
        writeAttribute(gridGroup, "dz", grid.dz);
        writeAttribute(gridGroup, "background_relative_permittivity",
                       geometry.background().relativePermittivity);
        writeAttribute(gridGroup, "background_relative_permeability",
                       geometry.background().relativePermeability);
        writeAttribute(gridGroup, "background_refractive_index",
                       geometry.background().refractiveIndex());
        writeStringAttribute(gridGroup, "field_layout",
                             "Yee staggered grid; x index is contiguous");

        std::vector<double> relativePermittivity(
            grid.nx * grid.ny * grid.nz, 0.0);
        std::vector<double> relativePermeability(
            grid.nx * grid.ny * grid.nz, 0.0);
        for (std::size_t k = 0; k < grid.nz; ++k) {
            for (std::size_t j = 0; j < grid.ny; ++j) {
                for (std::size_t i = 0; i < grid.nx; ++i) {
                    const double x = (static_cast<double>(i) + 0.5) * grid.dx;
                    const double y = (static_cast<double>(j) + 0.5) * grid.dy;
                    const double z = (static_cast<double>(k) + 0.5) * grid.dz;
                    const Material& material = geometry.materialAt(x, y, z);
                    const std::size_t id = grid.index(i, j, k,
                                                      grid.nx, grid.ny);
                    relativePermittivity[id] =
                        material.relativePermittivity;
                    relativePermeability[id] =
                        material.relativePermeability;
                }
            }
        }
        writeDataset(gridGroup, "relative_permittivity",
                     {grid.nz, grid.ny, grid.nx}, relativePermittivity);
        writeDataset(gridGroup, "relative_permeability",
                     {grid.nz, grid.ny, grid.nx}, relativePermeability);
    }

    void writeFrame(std::size_t step,
                    double time,
                    const Field3D& field,
                    const std::vector<Particle>& particles) {
        std::ostringstream name;
        name << "/steps/" << std::setw(8) << std::setfill('0') << step;
        const H5::Group group = file_.createGroup(name.str());
        writeAttribute(group, "step", step);
        writeAttribute(group, "time", time);
        writeAttribute(group, "particle_count", particles.size());

        writeDataset(group, "Ex", {grid_.nz + 1, grid_.ny + 1, grid_.nx},
                     field.Ex);
        writeDataset(group, "Ey", {grid_.nz + 1, grid_.ny, grid_.nx + 1},
                     field.Ey);
        writeDataset(group, "Ez", {grid_.nz, grid_.ny + 1, grid_.nx + 1},
                     field.Ez);
        writeDataset(group, "Bx", {grid_.nz, grid_.ny, grid_.nx + 1},
                     field.Bx);
        writeDataset(group, "By", {grid_.nz, grid_.ny + 1, grid_.nx},
                     field.By);
        writeDataset(group, "Bz", {grid_.nz + 1, grid_.ny, grid_.nx},
                     field.Bz);
        writeDataset(group, "Jx", {grid_.nz + 1, grid_.ny + 1, grid_.nx},
                     field.Jx);
        writeDataset(group, "Jy", {grid_.nz + 1, grid_.ny, grid_.nx + 1},
                     field.Jy);
        writeDataset(group, "Jz", {grid_.nz, grid_.ny + 1, grid_.nx + 1},
                     field.Jz);
        writeDataset(group, "rho",
                     {grid_.nz + 1, grid_.ny + 1, grid_.nx + 1},
                     field.rho);

        const std::vector<double> electricSlice =
            cellCenteredElectricMagnitudeSlice(field);
        writeDataset(group, "E_magnitude_xy",
                     {grid_.ny, grid_.nx}, electricSlice);
        writeStringAttribute(group, "E_magnitude_xy_plane",
                             "cell-centered z = nz/2");

        if (!particles.empty()) {
            std::vector<double> particleData;
            particleData.reserve(particles.size() * 8);
            for (const Particle& particle : particles) {
                particleData.insert(particleData.end(),
                                    {particle.x, particle.y, particle.z,
                                     particle.px, particle.py, particle.pz,
                                     particle.q, particle.m});
            }
            writeDataset(group, "particles", {particles.size(), 8},
                         particleData);
            writeStringAttribute(group, "particle_columns",
                                 "x,y,z,px,py,pz,q,m");
        }

        file_.flush(H5F_SCOPE_LOCAL);
    }

private:
    H5::H5File file_;
    const YeeGrid3D& grid_;

    static std::vector<hsize_t> toHdfDimensions(
        const std::vector<std::size_t>& dimensions) {
        std::vector<hsize_t> result;
        result.reserve(dimensions.size());
        for (const std::size_t dimension : dimensions) {
            result.push_back(static_cast<hsize_t>(dimension));
        }
        return result;
    }

    static void writeDataset(const H5::Group& group,
                             const std::string& name,
                             const std::vector<std::size_t>& dimensions,
                             const std::vector<double>& data) {
        std::size_t expectedSize = 1;
        for (const std::size_t dimension : dimensions) {
            expectedSize *= dimension;
        }
        if (expectedSize != data.size()) {
            throw std::runtime_error("HDF5 dataset size mismatch for " + name);
        }

        const std::vector<hsize_t> hdfDimensions =
            toHdfDimensions(dimensions);
        const H5::DataSpace dataSpace(
            static_cast<int>(hdfDimensions.size()), hdfDimensions.data());
        H5::DSetCreatPropList properties;
        std::vector<hsize_t> chunks = hdfDimensions;
        if (!chunks.empty()) {
            chunks[0] = std::min<hsize_t>(chunks[0], 4);
            if (chunks.size() > 1) {
                chunks[1] = std::min<hsize_t>(chunks[1], 32);
            }
            if (chunks.size() > 2) {
                chunks[2] = std::min<hsize_t>(chunks[2], 32);
            }
            properties.setChunk(static_cast<int>(chunks.size()), chunks.data());
            properties.setDeflate(4);
        }
        const H5::DataSet dataset = group.createDataSet(
            name, H5::PredType::NATIVE_DOUBLE, dataSpace, properties);
        dataset.write(data.data(), H5::PredType::NATIVE_DOUBLE);
    }

    static void writeAttribute(const H5::Group& group,
                               const std::string& name,
                               double value) {
        const H5::DataSpace scalar(H5S_SCALAR);
        const H5::Attribute attribute = group.createAttribute(
            name, H5::PredType::NATIVE_DOUBLE, scalar);
        attribute.write(H5::PredType::NATIVE_DOUBLE, &value);
    }

    static void writeAttribute(const H5::Group& group,
                               const std::string& name,
                               std::size_t value) {
        const H5::DataSpace scalar(H5S_SCALAR);
        const H5::Attribute attribute = group.createAttribute(
            name, H5::PredType::NATIVE_ULLONG, scalar);
        const auto converted = static_cast<unsigned long long>(value);
        attribute.write(H5::PredType::NATIVE_ULLONG, &converted);
    }

    static void writeStringAttribute(const H5::Group& group,
                                     const std::string& name,
                                     const std::string& value) {
        const H5::DataSpace scalar(H5S_SCALAR);
        H5::StrType stringType(H5::PredType::C_S1, H5T_VARIABLE);
        const H5::Attribute attribute =
            group.createAttribute(name, stringType, scalar);
        attribute.write(stringType, value);
    }

    std::vector<double> cellCenteredElectricMagnitudeSlice(
        const Field3D& field) const {
        const std::size_t k = grid_.nz / 2;
        std::vector<double> result(grid_.nx * grid_.ny, 0.0);
        for (std::size_t j = 0; j < grid_.ny; ++j) {
            for (std::size_t i = 0; i < grid_.nx; ++i) {
                const double ex = 0.25 *
                    (field.Ex[grid_.indexEx(i, j, k)] +
                     field.Ex[grid_.indexEx(i, j + 1, k)] +
                     field.Ex[grid_.indexEx(i, j, k + 1)] +
                     field.Ex[grid_.indexEx(i, j + 1, k + 1)]);
                const double ey = 0.25 *
                    (field.Ey[grid_.indexEy(i, j, k)] +
                     field.Ey[grid_.indexEy(i + 1, j, k)] +
                     field.Ey[grid_.indexEy(i, j, k + 1)] +
                     field.Ey[grid_.indexEy(i + 1, j, k + 1)]);
                const double ez = 0.25 *
                    (field.Ez[grid_.indexEz(i, j, k)] +
                     field.Ez[grid_.indexEz(i + 1, j, k)] +
                     field.Ez[grid_.indexEz(i, j + 1, k)] +
                     field.Ez[grid_.indexEz(i + 1, j + 1, k)]);
                result[i + grid_.nx * j] = std::sqrt(ex * ex + ey * ey +
                                                      ez * ez);
            }
        }
        return result;
    }
};
