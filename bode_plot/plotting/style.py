"""
plotting/style.py — Dark-theme plot styling (instrument-grade aesthetic).
"""

import matplotlib.pyplot as plt

# ── Colour palette ────────────────────────────────────────
BG     = "#0b0f14"
PANEL  = "#111820"
GRID   = "#1e2a35"
SPINE  = "#243040"
TXT    = "#cdd9e5"
DIMTXT = "#637a90"
BLUE   = "#4fa3e0"
RED    = "#e05c5c"
GREEN  = "#4ecb82"
YELLOW = "#e0b94f"
PURPLE = "#a585e0"
CYAN   = "#4ec8d4"


def apply_style() -> None:
    """Apply global matplotlib rcParams for dark theme."""
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    PANEL,
        "axes.edgecolor":    SPINE,
        "axes.labelcolor":   TXT,
        "axes.titlecolor":   TXT,
        "xtick.color":       DIMTXT,
        "ytick.color":       DIMTXT,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "text.color":        TXT,
        "grid.color":        GRID,
        "grid.linewidth":    0.5,
        "font.family":       "monospace",
        "font.size":         9,
        "axes.titlesize":    9.5,
        "axes.labelsize":    8.5,
        "axes.titlepad":     7,
        "legend.fontsize":   7.5,
        "legend.framealpha": 0.4,
        "legend.facecolor":  "#0d151f",
        "legend.edgecolor":  SPINE,
        "lines.antialiased": True,
    })


def style_ax(ax: plt.Axes) -> None:
    """Apply per-axes dark styling with grid."""
    ax.set_facecolor(PANEL)
    ax.spines[:].set_color(SPINE)
    ax.grid(True, which="both", alpha=1.0)
    ax.grid(True, which="minor", alpha=0.4, linewidth=0.3)
