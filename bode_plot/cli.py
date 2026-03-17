"""
cli.py — Command-line interface for BLDC bandwidth measurement.

Usage:
  python -m bode_plot                              # live measurement
  python -m bode_plot --demo                       # synthetic demo
  python -m bode_plot --demo --noisy               # demo with noise
  python -m bode_plot --monitor                    # real-time UDP monitor
  python -m bode_plot --reanalyze 260317_1430      # re-analyze saved data
  python -m bode_plot --compare A.npz B.npz        # compare two results
  python -m bode_plot --nyquist result.npz         # Nyquist plot
"""

import argparse
import os

from .config import CFG
from .output_manager import OutputManager
from .plotting import load_frf_from_npz, plot_comparison, plot_nyquist, run_monitor
from .export import export_csv, export_json
from .measurement import BandwidthMeasurement
from .demo import run_demo
from .reanalyze import run_reanalyze


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BLDC current controller bandwidth measurement"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run with synthetic 2nd-order plant (no hardware needed)",
    )
    parser.add_argument(
        "--signal", choices=["all", "chirp", "multisine", "step"],
        default="all",
        help="Signal type: 'all' runs sequential chirp→multisine→step "
             "(default). Single types also supported.",
    )
    parser.add_argument(
        "--compare", nargs=2, metavar="NPZ",
        help="Compare two saved .npz result files.",
    )
    parser.add_argument(
        "--nyquist", metavar="NPZ",
        help="Generate Nyquist plot from a saved .npz result file.",
    )
    parser.add_argument(
        "--monitor", action="store_true",
        help="Real-time UDP monitor (scrolling i_ref/i_meas plot).",
    )
    parser.add_argument(
        "--export-csv", metavar="NPZ",
        help="Export .npz result to CSV.",
    )
    parser.add_argument(
        "--export-json", metavar="NPZ",
        help="Export .npz result to JSON.",
    )
    parser.add_argument(
        "--noisy", action="store_true",
        help="(demo only) Add realistic BLDC noise and run adaptive pipeline.",
    )
    parser.add_argument(
        "--reanalyze", metavar="FOLDER",
        help="Re-analyze saved .npz data from a previous measurement folder. "
             "Example: --reanalyze 260317_1430",
    )
    args = parser.parse_args()

    cfg = CFG

    if args.monitor:
        run_monitor(cfg)
    elif args.reanalyze:
        run_reanalyze(args.reanalyze, signal_type=args.signal, cfg=cfg)
    elif args.nyquist:
        om  = OutputManager()
        frf = load_frf_from_npz(args.nyquist)
        plot_nyquist(frf, cfg, save_path=om.path("nyquist_result.png"))
        om.log_structure()
    elif args.export_csv:
        om  = OutputManager()
        frf = load_frf_from_npz(args.export_csv)
        base = os.path.splitext(os.path.basename(args.export_csv))[0]
        export_csv(frf, path=om.path(f"{base}.csv"))
        om.log_structure()
    elif args.export_json:
        om  = OutputManager()
        frf = load_frf_from_npz(args.export_json)
        base = os.path.splitext(os.path.basename(args.export_json))[0]
        export_json(frf, path=om.path(f"{base}.json"))
        om.log_structure()
    elif args.compare:
        om  = OutputManager()
        frf_a = load_frf_from_npz(args.compare[0])
        frf_b = load_frf_from_npz(args.compare[1])
        plot_comparison(frf_a, frf_b, cfg,
                        save_path=om.path("bandwidth_comparison.png"))
        om.log_structure()
    elif args.demo:
        run_demo(signal_type=args.signal, noisy=args.noisy)
    else:
        meas = BandwidthMeasurement(cfg, signal_type=args.signal)
        meas.run()


if __name__ == "__main__":
    main()
