"""Chapter 6 example using the gear manuscript's analytical contact solver."""
import argparse
from pathlib import Path
import sys

import matplotlib
import numpy as np
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BOOK = Path(__file__).resolve().parents[3]
SCALE = np.array([0.05, 1.0, 0.01, 1.0])
LOWER = np.array([0.0, 0.30, 0.0, 0.05])
UPPER = np.array([0.10, 0.95, 0.05, 0.50])
STARTS = np.array([[0.030, 0.65, 0.010, 0.35],
                   [0.015, 0.85, 0.020, 0.45],
                   [0.010, 0.50, 0.040, 0.48]])
SIGMA = 0.0001  # mm


def rms(values):
    return np.linalg.norm(values) / np.sqrt(values.size)


def fit(operator, target, starts, restricted=False):
    coordinates = slice(0, 2) if restricted else slice(None)

    def unpack(scaled):
        theta = scaled * SCALE[coordinates]
        return np.r_[theta, 0.0, 0.50] if restricted else theta

    def residual(scaled):
        return (operator.evaluate(unpack(scaled))["prediction"] - target) / SIGMA

    def tangent(scaled):
        jacobian = operator.evaluate(unpack(scaled), jacobian=True)["jacobian"]
        return jacobian[:, coordinates] * SCALE[coordinates] / SIGMA

    attempts = [
        least_squares(residual, start[coordinates] / SCALE[coordinates],
                      jac=tangent, method="trf",
                      bounds=(LOWER[coordinates] / SCALE[coordinates],
                              UPPER[coordinates] / SCALE[coordinates]),
                      max_nfev=1200, ftol=1e-11, xtol=1e-12, gtol=1e-9)
        for start in starts
    ]
    best = min(attempts, key=lambda attempt: attempt.cost)
    print(f"  fit RMS={1000 * SIGMA * rms(best.fun):.9g} um, "
          f"termination={[attempt.status for attempt in attempts]}, "
          f"theta={unpack(best.x)}")
    return unpack(best.x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=BOOK.parent /
                        "gears_inversion/deposit_v6/gear_inverse_identifiability/src",
                        help="src directory containing the archived gear operators")
    args = parser.parse_args()
    if not (args.source / "mechanics" / "batched_operator.py").is_file():
        parser.error("Set --source to the gear study's archived src directory.")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.source.resolve()))
    from geometry.involute import GearPair, InvoluteGear
    from geometry.microgeometry import TipRelief
    from mechanics.batched_operator import BatchedGapOperator
    from mechanics.compliance import GearContactModel

    pair = GearPair(InvoluteGear(24, 3, 20, 20), InvoluteGear(48, 3, 20, 20))
    relief = TipRelief()
    model = GearContactModel(pair, micro1=relief, compliance_variant="projected-energy")
    phase = np.arange(120) / 120
    check_phase = (np.arange(240) + 0.5) / 240
    training = BatchedGapOperator(model, 300, phi_positions=phase * (2 * np.pi / 24))
    checks = {torque: BatchedGapOperator(model, torque,
              phi_positions=check_phase * (2 * np.pi / 24)) for torque in (200, 300)}
    nominal = np.array([0.0, 0.75, 0.0, 0.50])
    print(f"Normal load={training.force:.6f} N, base radius={pair.g1.rb:.6f} mm")
    target = np.zeros(phase.size)
    print("Flat target, addendum-only:")
    tip_design = fit(training, target, STARTS, restricted=True)
    print("Flat target, combined relief:")
    design = fit(training, target, STARTS)
    for theta in [nominal, tip_design, design]:
        for torque, operator in [(300, training), *checks.items()]:
            result = operator.evaluate(theta)
            prediction = result["prediction"]
            print(f"  T={torque}, N={prediction.size}: "
                  f"RMS={1000 * rms(prediction):.7f} um, "
                  f"range={1000 * np.ptp(prediction):.7f} um, "
                  f"force imbalance={result['force_residual_N']:.2e} N, "
                  f"minimum gap={np.min(result['separations'][operator.valid]):.2e} mm")

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.65), layout="constrained")
    colors = {"nominal": "#51565e", "tip": "#a53d27", "combined": "#006773"}
    for name, theta, linestyle in [("nominal", nominal, ":"),
                                  ("tip", tip_design, "--"), ("combined", design, "-")]:
        values = checks[300].evaluate(theta)["prediction"]
        axes[0, 0].plot(check_phase, 1000 * values, linestyle, color=colors[name],
                        label={"nominal": "Unmodified", "tip": "Tip relief",
                               "combined": "Combined relief"}[name], lw=1.6)
    axes[0, 0].set(title="(a) Flat-target design at 300 N·m", ylabel="Centered TE [µm]")
    axes[0, 0].legend(frameon=False, fontsize=8)
    flank = np.linspace(0, 1, 1001)
    noise = SIGMA * np.random.default_rng(2026110100).normal(size=phase.size)
    rng = np.random.default_rng(20261101)
    restricted_starts = [np.array([0.020, 0.75]), *STARTS[:2, :2]]
    restricted_starts += [np.array([rng.uniform(0.001, 0.09),
                                     rng.uniform(0.31, 0.94)]) for _ in range(3)]
    tip_truth = np.array([0.020, 0.75, 0.0, 0.50])
    zero = training.evaluate(tip_truth)
    support_min = np.min(training.xi1[zero["forces"] > 0])
    print("Minimum loaded roll coordinate:", support_min)
    invisible = training.evaluate([0.020, 0.75, 0.005, 0.22])
    np.testing.assert_array_equal(invisible["delta"], zero["delta"])
    for column, sd in enumerate((0.25, 0.40)):
        truth = np.array([0.020, 0.75, 0.005, sd])
        evaluated = training.evaluate(truth, jacobian=True)
        reference = evaluated["prediction"]
        signal = reference - zero["prediction"]
        singular = np.linalg.svd(evaluated["jacobian"] * SCALE / SIGMA, compute_uv=False)
        print(f"Reference sd={sd:.2f}, singular values={singular}, "
              f"isolated RMS={1000 * rms(signal):.9g} um")
        restricted = fit(training, reference, restricted_starts, restricted=True)
        residual = reference - training.evaluate(restricted)["prediction"]
        print(f"  isolated contrast after refit: {1000 * rms(residual):.9g} um")
        observed = reference + noise
        estimate = fit(training, observed, STARTS)
        if column == 0:
            tip_estimate = estimate.copy()
            tip_estimate[2] = 0.0
            np.testing.assert_array_equal(training.evaluate(estimate)["delta"],
                                          training.evaluate(tip_estimate)["delta"])
        held_out = checks[200].evaluate(estimate)["prediction"]
        held_target = checks[200].evaluate(truth)["prediction"]
        print(f"  profile RMS={1000 * rms(relief.evaluate(flank, estimate) - relief.evaluate(flank, truth)):.9g} um, "
              f"200 Nm prediction RMS={1000 * rms(held_out - held_target):.9g} um")
        axes[0, 1].plot(phase, 1000 * signal, ("--", "-")[column],
                        color=("#a53d27", "#006773")[column], label=fr"$s_d={sd:.2f}$", lw=1.6)
        axis = axes[1, column]
        axis.plot(flank, 1000 * relief.evaluate(flank, truth), color="#51565e",
                  linestyle="--", lw=1.6, label="Generating profile")
        axis.plot(flank, 1000 * relief.evaluate(flank, estimate), color="#006773",
                  lw=1.6, label="Fit to noisy TE")
        axis.axvline(support_min, color="#a53d27",
                     linestyle=":", lw=1.2, label="Minimum loaded coordinate")
        axis.set(title=fr"({('c', 'd')[column]}) Recovery at $s_d={sd:.2f}$",
                 xlabel="Normalized roll coordinate", ylabel="Relief [µm]",
                 xlim=(0, 1))
        axis.legend(loc="upper left", frameon=True, facecolor="white",
                    edgecolor="none", framealpha=1, fontsize=7)
    axes[0, 1].set(title="(b) Isolated dedendum response", ylabel="TE difference [µm]")
    axes[0, 1].legend(frameon=False, fontsize=8)
    for axis in axes[0]:
        axis.set(xlabel="Mesh phase [fraction of pitch]", xlim=(0, 1))
    for axis in axes.flat:
        axis.grid(alpha=0.18)
    fig.savefig(BOOK / "figures" / "spur-gear-microgeometry-example.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
