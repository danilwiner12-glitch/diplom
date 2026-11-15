#!/usr/bin/env python3
"""Render an x-y electric-field slice from a PIC3D HDF5 snapshot."""

import argparse
import math

import h5py
import matplotlib.pyplot as plt
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="cherenkov.h5")
    parser.add_argument("--step", type=int,
                        help="step number; the last saved step is used by default")
    parser.add_argument("--output", default="cherenkov.png")
    parser.add_argument("--quantity", choices=("e", "bz"), default="e",
                        help="plot |E| or signed Bz on the central x-y plane")
    return parser.parse_args()


def select_step(file, requested_step):
    names = sorted(file["steps"].keys())
    if not names:
        raise RuntimeError("HDF5 file contains no snapshots")
    if requested_step is None:
        return names[-1]
    name = f"{requested_step:08d}"
    if name not in file["steps"]:
        raise RuntimeError(f"step {requested_step} is not present")
    return name


def main():
    arguments = parse_arguments()
    with h5py.File(arguments.input, "r") as file:
        step_name = select_step(file, arguments.step)
        snapshot = file["steps"][step_name]
        grid = file["grid"].attrs
        lx = float(grid["lx"])
        ly = float(grid["ly"])
        nz = int(grid["nz"])
        refractive_index = float(grid["background_refractive_index"])
        time = float(snapshot.attrs["time"])
        particles = snapshot["particles"][:] if "particles" in snapshot else None
        if arguments.quantity == "e":
            field = snapshot["E_magnitude_xy"][:]
        else:
            field = snapshot["Bz"][nz // 2, :, :]

    if arguments.quantity == "e":
        positive = field[field > 0.0]
        floor = np.max(positive) * 1.0e-8 if positive.size else 1.0e-30
        image = np.log10(np.maximum(field, floor))
        color_map = "inferno"
        color_label = "log10(|E| / (V/m))"
        color_limits = {}
    else:
        maximum = np.max(np.abs(field))
        floor = maximum * 1.0e-6 if maximum else 1.0e-30
        image = np.sign(field) * np.log10(1.0 + np.abs(field) / floor)
        limit = np.max(np.abs(image))
        color_map = "RdBu_r"
        color_label = "signed log scale of Bz"
        color_limits = {"vmin": -limit, "vmax": limit}

    figure, axes = plt.subplots(figsize=(11, 5), constrained_layout=True)
    plot = axes.imshow(
        image,
        origin="lower",
        extent=(0.0, lx * 1.0e3, 0.0, ly * 1.0e3),
        aspect="equal",
        cmap=color_map,
        **color_limits,
    )
    figure.colorbar(plot, ax=axes, label=color_label)

    if particles is not None and len(particles):
        particle = particles[0]
        x, y = particle[0] * 1.0e3, particle[1] * 1.0e3
        momentum = np.linalg.norm(particle[3:6])
        mass = particle[7]
        gamma = math.hypot(1.0, momentum / (mass * 299792458.0))
        beta = momentum / (gamma * mass * 299792458.0)
        axes.scatter([x], [y], color="cyan", s=28, label="macro-electron")
        if refractive_index * beta > 1.0:
            angle = math.acos(1.0 / (refractive_index * beta))
            slope = math.tan(angle)
            upper_distance = min(x, (ly * 1.0e3 - y) / slope)
            lower_distance = min(x, y / slope)
            axes.plot([x, x - upper_distance],
                      [y, y + slope * upper_distance],
                      "c--", linewidth=1.0)
            axes.plot([x, x - lower_distance],
                      [y, y - slope * lower_distance],
                      "c--", linewidth=1.0,
                      label=f"theory: {math.degrees(angle):.1f} deg")
        axes.legend(loc="upper right")

    axes.set_xlim(0.0, lx * 1.0e3)
    axes.set_ylim(0.0, ly * 1.0e3)
    axes.set_title(f"Cherenkov demo, step {int(step_name)}, t={time:.3e} s")
    axes.set_xlabel("x, mm")
    axes.set_ylabel("y, mm")
    figure.savefig(arguments.output, dpi=180)
    print(f"saved {arguments.output}")


if __name__ == "__main__":
    main()
