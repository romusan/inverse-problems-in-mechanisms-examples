"""Reproduce the direct and fixed-fixture synthesis examples in Chapter 3."""

from pathlib import Path

import matplotlib
import numpy as np
import scipy
from scipy.optimize import differential_evolution, least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GROUND = 60.0  # millimetres
NOMINAL = np.array([20.0, 70.0, 90.0, 45.0, 12.0])
LOWER = np.array([17.0, 67.0, 86.0, 38.0, -4.0])
UPPER = np.array([23.0, 73.0, 94.0, 52.0, 22.0])


def normal(vectors):
    return np.stack((-vectors[..., 1], vectors[..., 0]), axis=-1)


def forward(design, phase, ground=GROUND, branch=1):
    """Return crank tip, rocker joint, coupler tracer, and acute angle in rad."""
    crank, coupler, rocker, axial_offset, normal_offset = design
    crank_tip = crank * np.column_stack((np.cos(phase), np.sin(phase)))
    fixed_pivot = np.array([ground, 0.0])
    diagonal = fixed_pivot - crank_tip
    separation = np.linalg.norm(diagonal, axis=1)
    if np.any((separation <= abs(coupler - rocker)) |
              (separation >= coupler + rocker)):
        raise ValueError("The specified phase has no regular circle intersection.")
    axis = diagonal / separation[:, None]
    projection = (coupler**2 - rocker**2 + separation**2) / (2 * separation)
    height = np.sqrt(coupler**2 - projection**2)
    rocker_joint = crank_tip + projection[:, None] * axis
    rocker_joint += branch * height[:, None] * normal(axis)

    coupler_axis = (rocker_joint - crank_tip) / coupler
    rocker_axis = (rocker_joint - fixed_pivot) / rocker
    tracer = crank_tip + axial_offset * coupler_axis + normal_offset * normal(coupler_axis)
    cosine = np.sum(coupler_axis * rocker_axis, axis=1)
    transmission = np.arccos(np.clip(np.abs(cosine), 0.0, 1.0))
    return crank_tip, rocker_joint, tracer, transmission


def target(phase):
    """Synthetic curve from the adapted laboratory geometry with a known perturbation."""
    perturbation = np.column_stack((0.18 * np.cos(3 * phase),
                                    0.12 * np.sin(2 * phase)))
    return forward(NOMINAL, phase)[2] + perturbation


def report(label, design, phase):
    errors = np.linalg.norm(forward(design, phase)[2] - target(phase), axis=1)
    print(f"{label}: RMS = {np.sqrt(np.mean(errors**2)):.8f} mm, "
          f"peak = {np.max(errors):.8f} mm")


def direct_example(path):
    """Solve the course linkage at 60 degrees and verify its joint derivatives."""
    ground = 100.0
    design = [40.0, 80.0, 60.0, 0.0, 0.0]
    phase = np.deg2rad(60.0)
    speed = 4.0 * np.pi
    pivot = np.array([ground, 0.0])
    crank, joint, _, _ = forward(design, np.array([phase]), ground)
    _, opposite, _, _ = forward(design, np.array([phase]), ground, branch=-1)
    crank, joint, opposite = crank[0], joint[0], opposite[0]
    coupler = joint - crank
    rocker = joint - pivot
    matrix = np.column_stack((normal(coupler), -normal(rocker)))
    velocity_rhs = -normal(crank) * speed
    beta_rate, psi_rate = np.linalg.solve(matrix, velocity_rhs)
    acceleration_rhs = crank * speed**2 + coupler * beta_rate**2 - rocker * psi_rate**2
    beta_accel, psi_accel = np.linalg.solve(matrix, acceleration_rhs)
    velocity = normal(rocker) * psi_rate
    acceleration = normal(rocker) * psi_accel - rocker * psi_rate**2

    step = 1e-5
    phases = phase + speed * step * np.array([-1.0, 0.0, 1.0])
    _, (before, center, after), _, _ = forward(design, phases, ground)
    np.testing.assert_allclose((after - before) / (2 * step), velocity, atol=1e-4)
    np.testing.assert_allclose((after - 2 * center + before) / step**2,
                               acceleration, atol=3e-3)
    print("Direct: A, B+, B- [mm]:", crank, joint, opposite)
    for point in (joint, opposite):
        vectors = np.array([point - crank, point - pivot])
        print("Direct: beta, psi [deg]:",
              np.rad2deg(np.arctan2(vectors[:, 1], vectors[:, 0])))
    print("Direct: closure matrix and RHS vectors:", matrix, velocity_rhs, acceleration_rhs)
    print("Direct: angular rates [rad/s]:", beta_rate, psi_rate)
    print("Direct: angular accelerations [rad/s^2]:", beta_accel, psi_accel)
    print("Direct: B velocity and acceleration [mm/s, mm/s^2]:", velocity, acceleration)

    fig, axes = plt.subplots(figsize=(4.6, 2.65), layout="constrained")
    points = np.vstack(([0.0, 0.0], crank, joint, pivot))
    axes.plot(*points.T, color="#17365d", linewidth=1.7)
    axes.plot([0, ground], [0, 0], color="#607184", linewidth=1.7)
    axes.plot(*np.vstack((crank, pivot)).T, color="#607184",
              linestyle="--", linewidth=0.85)
    axes.scatter(*points.T, s=15, color="#17365d", zorder=3)
    arc = np.linspace(0.0, phase, 60)
    axes.plot(15 * np.cos(arc), 15 * np.sin(arc), color="#a54629", linewidth=1)
    axes.text(16, 8, r"$\phi$", color="#a54629", fontsize=11)
    for point, name, offset in zip(points, ("O", "A", "B", "D"),
                                   ((-10, 1), (-10, 5), (5, 4), (6, 0))):
        axes.annotate(name, point, xytext=offset, textcoords="offset points")
    axes.set(xlim=(-13, 116), ylim=(-7, 73),
             xlabel="x [mm]", ylabel="y [mm]", aspect="equal")
    axes.grid(color="#dfe5eb", linewidth=0.5)
    axes.set_axisbelow(True)
    fig.savefig(path)
    plt.close(fig)


def draw_example(path, design, fit_phase):
    phase = np.linspace(0.0, 2 * np.pi, 721)
    mechanism_phase = np.deg2rad(np.array([60.0]))
    crank, joint, tracer, _ = forward(design, mechanism_phase)
    points = np.vstack(([0.0, 0.0], crank[0], joint[0], [GROUND, 0.0]))
    coupler_axis = (joint[0] - crank[0]) / design[1]
    foot = crank[0] + design[3] * coupler_axis

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(6.35, 2.8), layout="constrained")
    mechanism, paths = axes
    loop = np.vstack((points, points[0]))
    mechanism.plot(*loop.T, color="#17365d", linewidth=1.65)
    mechanism.plot(*np.vstack((foot, tracer[0])).T, color="#a54629",
                   linestyle="--", linewidth=1.1)
    mechanism.scatter(*points.T, s=13, color="#17365d", zorder=3)
    mechanism.scatter(*tracer[0], s=15, color="#a54629", zorder=3)
    for point, name, offset in zip(
        np.vstack((points, tracer[0])), ("O", "A", "B", "D", "P"),
        ((-6, -11), (-7, 6), (5, 5), (5, -10), (-8, 6))
    ):
        mechanism.annotate(name, point, xytext=offset, textcoords="offset points")
    mechanism.set(xlim=(-10, 102), ylim=(-16, 97),
                  title=r"(a) Fitted linkage at $\phi=60^\circ$")

    paths.plot(*forward(design, phase)[2].T, color="#17365d", linewidth=1.45,
               label="Fitted path")
    paths.scatter(*target(fit_phase).T, s=12, facecolors="white",
                  edgecolors="#a54629", linewidths=0.75, zorder=3,
                  label="Target samples")
    paths.set_title("(b) Path in the fixed fixture")
    handles, labels = paths.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2,
               frameon=False, fontsize=8)
    for panel in axes:
        panel.set(xlabel="x [mm]", ylabel="y [mm]", aspect="equal")
        panel.grid(color="#dfe5eb", linewidth=0.5)
        panel.set_axisbelow(True)
    fig.savefig(path)
    plt.close(fig)


def main():
    phases = 2 * np.pi * np.arange(36) / 36
    check_phases = phases + np.pi / 36
    fit_target = target(phases)
    span = UPPER - LOWER

    def residual(z):
        return (forward(LOWER + span * z, phases)[2] - fit_target).ravel()

    def objective(z):
        return np.sum(residual(z)**2) / len(phases)

    result = differential_evolution(
        objective, [(0.0, 1.0)] * len(NOMINAL), strategy="best1bin",
        rng=26, popsize=8, maxiter=36, mutation=(0.5, 1.0),
        recombination=0.7, tol=0.0, atol=0.0, polish=False,
    )
    local = least_squares(
        residual, result.x, bounds=(0.0, 1.0), method="trf", jac="3-point",
        ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=200,
    )
    if not local.success:
        raise RuntimeError(local.message)

    de_design = LOWER + span * result.x
    fitted = LOWER + span * local.x
    print(f"NumPy {np.__version__}, SciPy {scipy.__version__}")
    print(f"DE objective calls: {result.nfev}")
    print("DE [a, b, r, xi, eta]:", de_design)
    print("Fit [a, b, r, xi, eta]:", fitted)
    print("First three target points:", fit_target[:3])
    report("Nominal on fit phases", NOMINAL, phases)
    report("After DE on fit phases", de_design, phases)
    report("After refinement on fit phases", fitted, phases)
    report("After refinement on shifted phases", fitted, check_phases)

    dense_phase = np.linspace(0.0, 2 * np.pi, 1441)
    crank, joint, tracer, transmission = forward(fitted, dense_phase)
    coupler_error = np.max(np.abs(np.linalg.norm(joint - crank, axis=1) - fitted[1]))
    rocker_error = np.max(np.abs(
        np.linalg.norm(joint - [GROUND, 0.0], axis=1) - fitted[2]
    ))
    return_error = max(np.linalg.norm(points[-1] - points[0])
                       for points in (crank, joint, tracer))
    separation_limits = (GROUND - fitted[0], GROUND + fitted[0])
    exact_minimum = np.min(forward(fitted, np.array([0.0, np.pi]))[3])
    print("Separation limits:", separation_limits)
    print("Circle intersection limits:", (abs(fitted[1] - fitted[2]),
                                           fitted[1] + fitted[2]))
    print(f"Min. transmission: {np.rad2deg(np.min(transmission)):.8f} deg (grid), "
          f"{np.rad2deg(exact_minimum):.8f} deg (endpoints)")
    print(f"Max. length error: {max(coupler_error, rocker_error):.3e} mm, "
          f"max. return error: {return_error:.3e} mm")

    book = Path(__file__).resolve().parents[3]
    draw_example(book / "figures" / "fourbar-synthesis-example.pdf", fitted, phases)
    direct_example(book / "figures" / "fourbar-direct-example.pdf")


if __name__ == "__main__":
    main()
