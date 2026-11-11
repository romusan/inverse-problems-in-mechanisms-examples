"""Replay the published Chapter 5 example with the C++ periodic evaluator."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MASS, STIFFNESS, DAMPING = 1.2, 15_000.0, 10.0
LIFT, SPEED, STEPS = 0.010, 100.0, 1200
ARCHIVED_JOB = "e7c4e7b84e3021bbf90a"
METRICS = ("total", "track", "smooth", "contact", "bounds", "slope", "lift",
           "offset", "max_error_m", "min_force_N", "cam_span_m", "cam_mean_m")


def cycloidal_lift(theta):
    """Sample the analytic target at the supplied knot angles."""
    def rise(u):
        return u - np.sin(2 * np.pi * u) / (2 * np.pi)

    u_rise = theta / (np.pi / 2)
    u_return = (theta - 3 * np.pi / 4) / (np.pi / 2)
    return LIFT * np.select(
        [theta < np.pi / 2, theta < 3 * np.pi / 4, theta < 5 * np.pi / 4],
        [rise(u_rise), 1.0, 1 - rise(u_return)],
        default=0.0,
    )


def check_metrics(result, expected):
    np.testing.assert_allclose([result[key] for key in METRICS],
                               [float(expected[key]) for key in METRICS],
                               rtol=0, atol=1e-8)


def replay(project):
    campaign = project / "revision_campaign"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(campaign))
    from objective import CamObjective

    record = campaign / "results_periodic16" / f"{ARCHIVED_JOB}.json"
    archived = json.loads(record.read_text(encoding="utf-8"))["payload"]
    if (archived["algorithm"], archived["seed"], archived["evaluation_mode"]) != (
            "multistart", 2049, "periodic_fixed_point"):
        raise ValueError("Unexpected archived design")
    selected = np.asarray(archived["controls"])
    if selected.size != 16:
        raise ValueError("The archived profile must have 16 independent controls")

    baseline_path = (project / "outputs" / "baseline_speed_20260802"
                     / "baseline_transient_vs_periodic.csv")
    with baseline_path.open(newline="", encoding="utf-8-sig") as stream:
        baseline_archive = {row["evaluation"]: row for row in csv.DictReader(stream)}
    baseline = cycloidal_lift(2 * np.pi * np.arange(12) / 12)

    def evaluator(omega=SPEED, periodic=True):
        return CamObjective(mass=MASS, stiffness=STIFFNESS, damping=DAMPING,
                            omega=omega, lift=LIFT, steps=STEPS, periodic=periodic)

    nominal = evaluator()
    baseline_metrics = nominal.metrics(baseline)
    selected_metrics = nominal.metrics(selected)
    check_metrics(baseline_metrics, baseline_archive["direct_periodic_fixed_point"])
    check_metrics(selected_metrics, archived["metrics"])

    transient_metrics = evaluator(periodic=False).metrics(baseline)
    check_metrics(transient_metrics, baseline_archive["five_turns_from_rest"])

    sweep_path = (project / "outputs" / "periodic16_robustness_20260802"
                  / "speed_sweep.csv")
    with sweep_path.open(newline="", encoding="utf-8-sig") as stream:
        sweep_archive = list(csv.DictReader(stream))
    speeds = np.array([float(row["omega"]) for row in sweep_archive])
    peak_mm = []
    for speed, archived_row in zip(speeds, sweep_archive):
        result = evaluator(speed).metrics(selected)
        check_metrics(result, archived_row)
        peak_mm.append(result["max_error_m"] * 1000)

    binary_path = Path(nominal.dll._name)
    print(f"C++ binary SHA-256: {hashlib.sha256(binary_path.read_bytes()).hexdigest()}")
    print(f"Archived inverse search evaluations: {archived['evaluations']}")
    for label, metrics in (("Periodic baseline", baseline_metrics),
                           ("Periodic selected design", selected_metrics),
                           ("Five-revolution baseline", transient_metrics)):
        print(f"{label}: J={metrics['total']:.9f}, "
              f"RMS={1000 * LIFT * np.sqrt(metrics['track']):.6f} mm, "
              f"peak={1000 * metrics['max_error_m']:.6f} mm, "
              f"min surrogate={metrics['min_force_N']:.6f} N, "
              f"input span={1000 * metrics['cam_span_m']:.6f} mm")
    print("Controls [mm]:", ", ".join(f"{h * 1000:.6f}" for h in selected))
    print("Fixed-profile speed sweep verified at", len(speeds), "speeds.")
    return (nominal.trace(baseline, 2400), nominal.trace(selected, 2400),
            speeds, np.array(peak_mm))


def draw(path, baseline, selected, speeds, peak_mm):
    def close(values):
        return 1000 * np.r_[values[-1], values]

    phase_deg = 360 * np.arange(len(selected) + 1) / len(selected)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, (motion, speed) = plt.subplots(1, 2, figsize=(6.25, 2.7),
                                       layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.14, 1, 0.86))
    motion.plot(phase_deg, close(selected[:, 2]), color="#28313c", linewidth=1.4,
                label="Target")
    motion.plot(phase_deg, close(baseline[:, 3]), color="#a54a30",
                linestyle=":", linewidth=1.55, label="Baseline follower")
    motion.plot(phase_deg, close(selected[:, 1]), color="#6f7786",
                linestyle="--", linewidth=1.2, label="Optimized input")
    motion.plot(phase_deg, close(selected[:, 3]), color="#00747b", linewidth=1.5,
                label="Optimized follower")
    motion.set(title="(a) Periodic motion at 100 rad/s", xlabel="Cam angle [°]",
               ylabel="Displacement [mm]", xlim=(0, 360),
               xticks=[0, 90, 180, 270, 360])

    natural = np.sqrt(STIFFNESS / MASS)
    speed.plot(speeds, peak_mm, color="#00747b", marker="o", markersize=2.9,
               linewidth=1.4, label="Fixed optimized profile")
    speed.axvline(SPEED, color="#6f7786", linestyle=":", linewidth=1.2,
                  label="Design speed")
    for harmonic, label in ((2, r"$\omega_n/2$"), (1, r"$\omega_n$")):
        crossing = natural / harmonic
        speed.axvline(crossing, color="#a54a30", linestyle="--", linewidth=0.9)
        speed.text(crossing + 1.7, 17.6, label, color="#8a412d", fontsize=8)
    speed.set(title="(b) Speed sensitivity", xlabel="Shaft speed [rad/s]",
              ylabel="Peak tracking error [mm]", ylim=(0, 19), xlim=(38, 142))
    for panel, center in ((motion, 0.25), (speed, 0.75)):
        handles, labels = panel.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(center, 0),
                   ncols=2, frameon=False, fontsize=7.1)
        panel.grid(color="#dfe5eb", linewidth=0.5)
        panel.set_axisbelow(True)
    fig.savefig(path)
    plt.close(fig)


def main():
    book = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=book.parent / "paper_BL_TASS_CAM")
    args = parser.parse_args()
    draw(book / "figures" / "cam-periodic-inverse-example.pdf", *replay(args.project.resolve()))


if __name__ == "__main__":
    main()
