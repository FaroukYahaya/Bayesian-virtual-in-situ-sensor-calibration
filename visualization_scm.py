#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Farouk Yahaya
Contact: faroya2011@gmail.com
Date: 2025-10-04

Title: SCM Visualization
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Disable LaTeX and set simple fonts
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'sans-serif'


def plot_caseA_summary(plot_df, out_png, used_tod: bool):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(plot_df.index, plot_df["faulty"], label="Faulty", color='red', linewidth=1.5, alpha=0.7)
    ax.plot(plot_df.index, plot_df["calibrated"], label="Calibrated (Case A)", color='green', linewidth=2, alpha=0.8)
    ax.plot(plot_df.index, plot_df["reference"], label=("ToD surrogate" if used_tod else "Reference"),
            color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_title("Case A: Reference-based calibration")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temp (C)")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_caseB_summary(plot_df, out_png, tau_star: str | None):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(plot_df.index, plot_df.get("faulty"), label="Faulty", color='red', linewidth=1.5, alpha=0.7)
    ax.plot(plot_df.index, plot_df.get("calibrated"), label="Calibrated (Case B)", color='green', linewidth=2, alpha=0.8)
    ax.plot(plot_df.index, plot_df.get("tod_mean"), label="Time-of-day mean",
            color='blue', linestyle='--', linewidth=1.5, alpha=0.7)

    if tau_star:
        try:
            import pandas as pd
            tau_dt = pd.to_datetime(tau_star)
            ax.axvline(tau_dt, linestyle=":", color='orange', linewidth=3, alpha=0.8, label="tau*")

            # Add marker at detection point
            if tau_dt in plot_df.index:
                y_val = plot_df.loc[tau_dt, "faulty"]
            else:
                idx = plot_df.index.get_indexer([tau_dt], method='nearest')[0]
                y_val = plot_df.iloc[idx]["faulty"]

            ax.plot(tau_dt, y_val, marker='D', color='orange', markersize=10,
                    markeredgewidth=1.5, markeredgecolor='black', markerfacecolor='orange')
        except Exception:
            pass

    ax.set_title("Case B: VIC drift correction")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temp (C)")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
