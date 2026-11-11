"""Compare verified candidates under explicit Chapter 7 scenarios."""
import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
from scipy.stats import beta

BOOK = Path(__file__).resolve().parents[3]


def load_example(category, module):
    sys.path.insert(0, str(BOOK / "chapters" / category / "examples"))
    return importlib.import_module(module)


def linkage_scenarios(project):
    archive = project / "envio_Applied_Sciences/revision_2026/12_campanas_corregidas"
    sys.path.insert(0, str(archive / "source"))
    example = load_example("linkages", "fef_graph_comparison")
    from campaign import target_points
    from fast_geometry import resample, simulate
    from revision_engine import FAMILIES, evaluate, physical_check

    target, _ = target_points("medium_hard", n=120)
    phase = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    family = "wattI"
    limits = np.array([1.0, 2.0])
    for budget, key in [(10_000, "best_rms"), (20_000, "best_strict")]:
        record = example.read_checkpoint(archive / "results", family, budget)[key]
        design = np.asarray(record["x"])
        nominal = evaluate(family, design, target, phase)
        raw = nominal["P_eval"]
        registered = nominal["P_aligned"]
        source, destination = raw - raw.mean(axis=0), registered - registered.mean(axis=0)
        u, _, vt = np.linalg.svd(source.T @ destination)
        rotation = u @ vt
        assert np.linalg.det(rotation) > 0
        translation = registered.mean(axis=0) - raw.mean(axis=0) @ rotation
        np.testing.assert_allclose(raw @ rotation + translation, registered, atol=1e-10)
        worst_fit = 0.0
        worst_angle = np.inf
        for offset in (0.0, -0.10, 0.10):
            actual = design.copy()
            actual[0] += offset
            valid, joints, _ = simulate(FAMILIES.index(family), actual, phase)
            cycle = physical_check(family, actual, n=1920)
            if not valid or not cycle["motion_valid"]:
                raise RuntimeError(f"Cycle failure for {budget} calls at offset {offset}")
            tracer = resample(joints[:, 4], len(target)) @ rotation + translation
            distances = np.linalg.norm(tracer - nominal["target_shifted"], axis=1)
            metrics = np.array([np.linalg.norm(distances) / np.sqrt(distances.size),
                                distances.max()])
            if offset == 0.0:
                np.testing.assert_allclose(metrics, [record["rms_mm"], record["max_mm"]],
                                           atol=1e-8, rtol=0)
            angle = cycle["min_angle_deg"]
            worst_fit = max(worst_fit, max(metrics / limits))
            worst_angle = min(worst_angle, angle)
            print(f"CD-G {budget}, offset {offset:+.2f}: RMS {metrics[0]:.8f}, "
                  f"peak {metrics[1]:.8f} mm, angle {angle:.8f} deg")
        print(f"  requirement ratio {max(worst_fit, 30 / worst_angle):.8f}")


def cam_scenarios(project):
    example = load_example("cams", "periodic_cam_example")
    campaign = project / "revision_campaign"
    sys.path.insert(0, str(campaign))
    from objective import CamObjective

    path = campaign / "results_periodic16" / f"{example.ARCHIVED_JOB}.json"
    archived = json.loads(path.read_text(encoding="utf-8"))["payload"]
    controls = [example.cycloidal_lift(2 * np.pi * np.arange(12) / 12),
                np.asarray(archived["controls"])]
    path = project / "outputs/periodic16_robustness_20260802"
    with (path / "speed_sweep.csv").open(newline="", encoding="utf-8-sig") as file:
        expected = {float(row["omega"]): row for row in csv.DictReader(file)}
    errors = []
    for speed in (90, 95, 100, 105, 110, 115):
        evaluator = CamObjective(mass=example.MASS, stiffness=example.STIFFNESS,
                                 damping=example.DAMPING, omega=speed, lift=example.LIFT,
                                 steps=example.STEPS, periodic=True)
        metrics = [evaluator.metrics(profile) for profile in controls]
        example.check_metrics(metrics[1], expected[speed])
        errors.append([1000 * row["max_error_m"] for row in metrics])
        print(f"Cam {speed} rad/s, peak [mm]: {errors[-1]}")
    print("  requirement ratios on narrow and extended grids:",
          np.max(errors[:4], axis=0) / 6.0, np.max(errors, axis=0) / 6.0)
    with (path / "monte_carlo_pm10.csv").open(newline="", encoding="utf-8-sig") as file:
        peak = np.array([float(row["max_error_m"]) for row in csv.DictReader(file)])
    count = np.count_nonzero(peak > 0.010)
    assert (count, peak.size) == (73, 1000)
    interval = (beta.ppf(0.025, count, peak.size - count + 1),
                beta.ppf(0.975, count + 1, peak.size - count))
    print(f"  archived 10 mm exceedance: {count}/{peak.size}; 95% CI {interval}")


def gear_scenarios(project):
    sys.path.insert(0, str(project / "deposit_v6/gear_inverse_identifiability/src"))
    example = load_example("gears", "spur_gear_microgeometry")
    from geometry.involute import GearPair, InvoluteGear
    from geometry.microgeometry import TipRelief
    from mechanics.batched_operator import BatchedGapOperator
    from mechanics.compliance import GearContactModel

    pair = GearPair(InvoluteGear(24, 3, 20, 20), InvoluteGear(48, 3, 20, 20))
    models = {variant: GearContactModel(pair, micro1=TipRelief(), compliance_variant=variant)
              for variant in ("projected-energy", "shear-G")}
    training = BatchedGapOperator(models["projected-energy"], 300,
                                  phi_positions=np.arange(120) / 120 * (2 * np.pi / 24))
    target = np.zeros(120)
    designs = [np.array([0.0, 0.75, 0.0, 0.50]),
               example.fit(training, target, example.STARTS, restricted=True),
               example.fit(training, target, example.STARTS)]
    check_phase = (np.arange(240) + 0.5) / 240 * (2 * np.pi / 24)
    ratios = np.zeros(len(designs))
    for variant, model in models.items():
        for torque in (200, 300):
            check = BatchedGapOperator(model, torque, phi_positions=check_phase)
            for i, design in enumerate(designs):
                prediction = 1000 * check.evaluate(design)["prediction"]
                rms, peak_to_peak = example.rms(prediction), np.ptp(prediction)
                ratios[i] = max(ratios[i], rms / 2.5, peak_to_peak / 8.0)
                print(f"Gear {variant}, {torque} N m, design {i + 1}: "
                      f"RMS {rms:.8f}, peak-to-peak {peak_to_peak:.8f} um")
    print("  requirement ratios:", ratios)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research", type=Path, default=BOOK.parent,
                        help="parent directory of the three study projects")
    root = parser.parse_args().research.resolve()
    sys.dont_write_bytecode = True
    with TemporaryDirectory(prefix="mechanisms_chapter7_") as cache:
        os.environ["NUMBA_CACHE_DIR"] = cache
        linkage_scenarios(root / "FEF-Graph")
    cam_scenarios(root / "paper_BL_TASS_CAM")
    gear_scenarios(root / "gears_inversion")


if __name__ == "__main__":
    main()
