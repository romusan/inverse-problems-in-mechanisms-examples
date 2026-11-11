"""Recheck the archived Chapter 4 comparison and regenerate its figure."""

import argparse
import csv
import json
import os
from pathlib import Path
from statistics import median
import sys
from tempfile import TemporaryDirectory

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODELS = {
    "fourbar": "F4",
    "slidercrank": "SC",
    "wattI": "CD-G",
    "wattII": "RC-G",
    "stephensonI": "A-O4",
    "stephensonII": "B-O2",
    "stephensonIII": "C-O4",
    "gearedFiveBar": "G5",
}
SELECTED = ("fourbar", "wattI", "gearedFiveBar")


def report_aggregates(revision):
    path = revision / "13_cierre_editorial" / "results" / "per_seed_v2.csv"
    with path.open(newline="", encoding="utf-8") as source:
        rows = [row for row in csv.DictReader(source)
                if row["target"] == "medium_hard" and row["method"] == "hybrid"
                and int(row["budget"]) == 20_000]
    for profile in ("compact", "relaxed"):
        for family, model in MODELS.items():
            cell = [row for row in rows if row["profile"] == profile
                    and row["family"] == family]
            if sorted(int(row["seed"]) for row in cell) != list(range(1, 31)):
                raise ValueError(f"Incomplete {profile} cell for {model}")
            counts = [sum(row[key] == "True" for row in cell)
                      for key in ("strict", "strict_motion", "strict_quality30")]
            rms = median(float(row["rms"]) for row in cell)
            print(f"{profile:7} {model:4}  median RMS {rms:.3f}  A/M/Q {counts}")


def read_checkpoint(results, family, budget):
    name = f"medium_hard_compact_hybrid_{family}_seed001_b40000.json"
    run = json.loads((results / name).read_text(encoding="utf-8"))
    return next(item for item in run["checkpoints"] if item["simulations"] == budget)


def check_examples(campaign_dir):
    from campaign import target_points
    from fast_geometry import simulate
    from revision_engine import FAMILIES, acute, configure, evaluate, physical_check

    configure("compact")
    target, _ = target_points("medium_hard", n=120)
    fit_grid = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    cycle_grid = np.linspace(0, 2 * np.pi, 1921)
    paths = {}
    angles = None

    for family, budget in (("wattI", 10_000), *((f, 20_000) for f in SELECTED)):
        checkpoint = read_checkpoint(campaign_dir / "results", family, budget)
        key = "best_strict" if family == "wattI" and budget == 20_000 else "best_rms"
        saved = checkpoint[key]
        design = np.asarray(saved["x"])
        fit = evaluate(family, design, target, fit_grid)
        cycle = physical_check(family, design, n=1920)
        np.testing.assert_allclose([fit["rms_mm"], fit["max_mm"]],
                                   [saved["rms_mm"], saved["max_mm"]],
                                   rtol=0, atol=1e-8)
        np.testing.assert_allclose(cycle["min_acute_angles_deg"],
                                   saved["physical"]["min_acute_angles_deg"],
                                   rtol=0, atol=1e-8)
        for flag in ("motion_valid", "quality30"):
            if cycle[flag] != saved["physical"][flag]:
                raise ValueError(f"{flag} changed for {family}, {budget} calls")

        print(f"{MODELS[family]} {budget}: RMS {fit['rms_mm']:.6f}, "
              f"peak {fit['max_mm']:.6f}, endpoint {cycle['state_seam_mm']:.2e}, "
              f"link {cycle['link_residual_mm']:.2e}, "
              f"min angle {cycle['min_angle_deg']:.6f}, "
              f"cycle {cycle['motion_valid']}, quality {cycle['quality30']}")
        if budget == 20_000:
            paths[family] = np.roll(fit["P_aligned"], -fit["shift"], axis=0)
        if family == "wattI" and budget == 20_000:
            valid, points, _ = simulate(FAMILIES.index(family), design, cycle_grid)
            if not valid:
                raise ValueError("CD-G closure failed on the cycle grid")
            a, b, c, d, _ = np.moveaxis(points, 1, 0)
            o4 = np.array([design[0], 0.0])
            g = design[6:8]
            angles = acute(b - a, o4 - b), acute(d - c, g - d)
            np.testing.assert_allclose(np.min(angles, axis=1),
                                       cycle["min_acute_angles_deg"], atol=1e-8)

    return target, paths, np.rad2deg(cycle_grid), angles


def draw(path, target, fitted, degrees, angles):
    def closed(points):
        return np.vstack((points, points[0]))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, (shape, cycle) = plt.subplots(1, 2, figsize=(6.25, 2.68),
                                      layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.12, 1, 0.88))
    shape.plot(*closed(target).T, label="Target", color="#343d48",
               linewidth=1.8, zorder=2)
    for family, color, style in (("fourbar", "#636d79", ":"),
                                 ("wattI", "#00747b", "--"),
                                 ("gearedFiveBar", "#b45624", "-.")):
        shape.plot(*closed(fitted[family]).T, color=color, linestyle=style,
                   label=MODELS[family], linewidth=1.35, zorder=3)
    shape.set(title="(a) Registered tracer paths", xlabel="x [mm]", ylabel="y [mm]",
              aspect="equal")

    cycle.plot(degrees, angles[0], label="First loop", color="#00747b")
    cycle.plot(degrees, angles[1], label="Second dyad", color="#b45624",
               linestyle="--")
    cycle.axhline(30.0, color="#343d48", linestyle=":", label="30° screen")
    cycle.set(title="(b) CD-G transmission", xlabel="Input angle [°]",
              ylabel="Acute angle [°]", xlim=(0, 360),
              xticks=[0, 90, 180, 270, 360])
    for panel, center, columns in ((shape, 0.25, 4), (cycle, 0.75, 3)):
        handles, labels = panel.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center",
                   bbox_to_anchor=(center, 0.01), ncols=columns,
                   frameon=False, fontsize=7.3)
        panel.grid(color="#dfe5eb", linewidth=0.5)
        panel.set_axisbelow(True)

    fig.savefig(path)
    plt.close(fig)


def main():
    book = Path(__file__).resolve().parents[3]
    default_revision = (book.parent / "FEF-Graph" / "envio_Applied_Sciences"
                        / "revision_2026")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", type=Path, default=default_revision)
    revision = parser.parse_args().revision.resolve()
    campaign_dir = revision / "12_campanas_corregidas"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(campaign_dir / "source"))
    report_aggregates(revision)
    with TemporaryDirectory(prefix="fef_chapter4_") as cache:
        os.environ["NUMBA_CACHE_DIR"] = cache
        values = check_examples(campaign_dir)
        draw(book / "figures" / "fef-graph-cycle-comparison.pdf", *values)


if __name__ == "__main__":
    main()
