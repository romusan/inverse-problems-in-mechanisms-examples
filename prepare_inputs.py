"""Fetch the pinned public study inputs needed by the book examples."""

from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
FEF_REVISION = "envio_Applied_Sciences/revision_2026"
FEF_CAMPAIGN = f"{FEF_REVISION}/12_campanas_corregidas"
GEAR_PREFIX = (
    "reproducibility/reproducibility_v20/revision_19_2026-11-01/"
    "reproducibility_v19/"
)
GEAR_SOURCE = "deposit_v6/gear_inverse_identifiability/src"


def checkout_inputs(repo, commit, folder, prefix, paths):
    """Copy the requested files from a sparse checkout of the cited commit."""
    print(f"Fetching {repo} at {commit[:8]}", flush=True)
    with TemporaryDirectory(prefix="book-inputs-") as work:
        checkout = Path(work)

        def git(*args, **options):
            subprocess.run(
                ["git", "-c", "core.longpaths=true", "-c", "core.autocrlf=false", *args],
                cwd=checkout, check=True, **options,
            )

        git("init", "-q")
        git("remote", "add", "origin", f"https://github.com/romusan/{repo}.git")
        git("fetch", "--depth=1", "--filter=blob:none", "origin", commit)
        patterns = "".join(f"/{prefix}{path}\n" for path in paths)
        git("sparse-checkout", "set", "--no-cone", "--stdin", input=patterns, text=True)
        git("checkout", "--quiet", "--detach", commit)

        for path in paths:
            source = checkout / f"{prefix}{path}"
            destination = INPUTS / folder / path
            if source.is_dir():
                shutil.copytree(source, destination, dirs_exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)


def main():
    checkout_inputs(
        "fef-graph-capability-benchmark", "0d772506e83a787b9e0b7793cd8475361ab9a454",
        "FEF-Graph", "",
        [
            f"{FEF_CAMPAIGN}/source/{name}.py"
            for name in ("campaign", "fast_geometry", "historical_engine", "revision_engine")
        ] + [
            f"{FEF_CAMPAIGN}/results/medium_hard_compact_hybrid_{family}_seed001_b40000.json"
            for family in ("fourbar", "wattI", "gearedFiveBar")
        ] + [f"{FEF_REVISION}/13_cierre_editorial/results/per_seed_v2.csv"],
    )
    checkout_inputs(
        "python", "13c84e315790bc1469b1ab6d2214f0ff3ecd4c66",
        "paper_BL_TASS_CAM", "rineng_113520/",
        [
            "revision_campaign/objective.py",
            "revision_campaign/cpp/cam_objective.cpp",
            "revision_campaign/results_periodic16/e7c4e7b84e3021bbf90a.json",
            "outputs/baseline_speed_20260802/baseline_transient_vs_periodic.csv",
            "outputs/periodic16_robustness_20260802/speed_sweep.csv",
            "outputs/periodic16_robustness_20260802/monte_carlo_pm10.csv",
        ],
    )
    checkout_inputs(
        "gears-inversion-reproducibility", "956fac28cd766bc53deea978d7334bc497d2e490",
        "gears_inversion", GEAR_PREFIX, [f"{GEAR_SOURCE}/"],
    )
    print(f"Study inputs are ready in {INPUTS}")


if __name__ == "__main__":
    main()
