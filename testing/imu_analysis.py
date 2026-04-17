# -*- coding: utf-8 -*-
"""
IMU Fall Detection Dataset Analysis
====================================
Compares real sensor data (S23T34R01 - Task 34: Backward fall while walking, slip)
with MuJoCo-simulated IMU data.

Files:
  - Simulated : fall_backward_walking_age32_20260408_144148.csv
  - Original  : S23T34R01.csv
  - Labels    : SA23_label.xlsx
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ---------------------------------------------
#  CONFIGURE PATHS  (edit if needed)
# ---------------------------------------------
BASE = Path("/mnt/hdd16T/ToqeerHomeBackup/mujoco_project/testing")
SIM_RAW   = BASE / "fall_backward_walking_age32_20260408_144148.csv"
ORIG_FILE = BASE / "S23T34R01.csv"
LABEL_FILE = BASE / "SA23_label.xlsx"

TASK_ID = 34          # F15 (34) = Backward fall while walking caused by a slip
TRIAL_ID = 1

OUTPUT_DIR = BASE     # where to save cleaned simulated CSV and plots
# ---------------------------------------------


# -----------------------------------------------------------
#  STEP 1 - LOAD & RESHAPE SIMULATED FILE
# -----------------------------------------------------------

def load_simulated(path: Path) -> pd.DataFrame:
    """
    Read the MuJoCo-generated CSV (skips all # comment metadata lines).
    Keeps only the IMU-relevant columns in a canonical order and
    renames them to match the original dataset convention.
    """
    df = pd.read_csv(path, comment="#")

    # -- columns to KEEP from the 27-column simulated file --------------
    #   timestamp  ? maps to time axis (seconds at 100 Hz)
    #   accel_x/y/z ? accelerometer  (m/s²)
    #   gyro_x/y/z  ? gyroscope      (rad/s)
    #   fall_detected ? ground-truth fall flag from simulator
    KEEP = ["timestamp", "accel_x", "accel_y", "accel_z",
            "gyro_x", "gyro_y", "gyro_z", "fall_detected"]

    df = df[KEEP].copy()

    # -- rename to a clean, stable schema -------------------------------
    df.rename(columns={
        "timestamp"    : "time_s",
        "accel_x"      : "AccX_ms2",
        "accel_y"      : "AccY_ms2",
        "accel_z"      : "AccZ_ms2",
        "gyro_x"       : "GyrX_rads",
        "gyro_y"       : "GyrY_rads",
        "gyro_z"       : "GyrZ_rads",
        "fall_detected": "fall_flag",
    }, inplace=True)

    # -- derived columns useful for analysis ----------------------------
    df["accel_mag_ms2"] = np.sqrt(df["AccX_ms2"]**2 +
                                   df["AccY_ms2"]**2 +
                                   df["AccZ_ms2"]**2)
    df["gyro_mag_dps"]  = np.degrees(
        np.sqrt(df["GyrX_rads"]**2 +
                df["GyrY_rads"]**2 +
                df["GyrZ_rads"]**2))

    # canonical column order
    df = df[["time_s",
             "AccX_ms2", "AccY_ms2", "AccZ_ms2", "accel_mag_ms2",
             "GyrX_rads", "GyrY_rads", "GyrZ_rads", "gyro_mag_dps",
             "fall_flag"]]

    print(f"[SIM]  Loaded {len(df)} rows  |  columns: {df.columns.tolist()}")
    return df


# -----------------------------------------------------------
#  STEP 2 - LOAD ORIGINAL DATASET + LABELS
# -----------------------------------------------------------

def load_original(path: Path, label_path: Path,
                  task_id: int = 34, trial_id: int = 1) -> pd.DataFrame:
    """
    Load the original S23 IMU recording and attach fall-onset / impact
    frame info from the label spreadsheet.

    Original units (SisFall convention):
        AccX/Y/Z  : mg   (milligravity, 1 mg = 0.00981 m/s²)
        GyrX/Y/Z  : mdps (milli-degrees/s)
    """
    df = pd.read_csv(path)

    # -- unit conversion -------------------------------------------------
    G = 9.81  # m/s²
    for ax in ["AccX", "AccY", "AccZ"]:
        df[f"{ax}_ms2"] = df[ax] * 0.001 * G   # mg ? m/s²

    for gx in ["GyrX", "GyrY", "GyrZ"]:
        df[f"{gx}_rads"] = np.radians(df[gx] * 0.001)  # mdps ? rad/s

    df["accel_mag_ms2"] = np.sqrt(df["AccX_ms2"]**2 +
                                   df["AccY_ms2"]**2 +
                                   df["AccZ_ms2"]**2)
    df["gyro_mag_dps"] = np.sqrt(df["GyrX"]**2 +
                                  df["GyrY"]**2 +
                                  df["GyrZ"]**2) * 0.001  # mdps ? dps

    # -- look up fall frames from label file -----------------------------
    labels = pd.read_excel(label_path)
    labels.columns = ["TaskCode", "Description", "TrialID",
                       "Fall_onset_frame", "Fall_impact_frame"]

    # Task codes look like "F15 (34)" - extract numeric ID
    labels["task_id_num"] = labels["TaskCode"].str.extract(r"\((\d+)\)").astype(int)
    row = labels[(labels["task_id_num"] == task_id) &
                 (labels["TrialID"] == trial_id)]

    if row.empty:
        raise ValueError(f"Task {task_id} Trial {trial_id} not found in labels.")

    onset_frame  = int(row["Fall_onset_frame"].iloc[0])
    impact_frame = int(row["Fall_impact_frame"].iloc[0])
    print(f"[ORIG] Task {task_id} | onset frame={onset_frame}, "
          f"impact frame={impact_frame}")

    # -- time axis: derive from FrameCounter + 100 Hz assumption ---------
    df["time_s"] = df["FrameCounter"] / 100.0

    # -- fall flag column aligned with simulated -------------------------
    df["fall_flag"] = 0
    df.loc[(df["FrameCounter"] >= onset_frame) &
           (df["FrameCounter"] <= impact_frame), "fall_flag"] = 1

    # -- keep only canonical columns -------------------------------------
    df = df[["time_s", "FrameCounter",
             "AccX_ms2", "AccY_ms2", "AccZ_ms2", "accel_mag_ms2",
             "GyrX_rads", "GyrY_rads", "GyrZ_rads", "gyro_mag_dps",
             "EulerX", "EulerY", "EulerZ",
             "fall_flag"]].copy()

    print(f"[ORIG] Loaded {len(df)} rows  |  fall flag rows: "
          f"{df['fall_flag'].sum()}")
    return df, onset_frame, impact_frame


# -----------------------------------------------------------
#  STEP 3 - SAVE CLEANED SIMULATED CSV
# -----------------------------------------------------------

def save_cleaned_sim(df_sim: pd.DataFrame, out_dir: Path):
    out_path = out_dir / "sim_cleaned_imu.csv"
    df_sim.to_csv(out_path, index=False)
    print(f"\n[SAVED] Cleaned simulated CSV ? {out_path}")
    print(f"        Columns : {df_sim.columns.tolist()}")
    print(f"        Shape   : {df_sim.shape}")
    return out_path


# -----------------------------------------------------------
#  STEP 4 - COMPARISON PLOTS
# -----------------------------------------------------------

ORIG_COLOR = "#1f77b4"   # blue
SIM_COLOR  = "#ff7f0e"   # orange
FALL_ALPHA = 0.18


def _shade_fall(ax, df, time_col="time_s", flag_col="fall_flag", color="red"):
    """Shade the fall window on an axes."""
    in_fall = False
    t_start = None
    for _, row in df.iterrows():
        if row[flag_col] == 1 and not in_fall:
            t_start = row[time_col]
            in_fall = True
        elif row[flag_col] == 0 and in_fall:
            ax.axvspan(t_start, row[time_col], color=color,
                       alpha=FALL_ALPHA, label="fall window")
            in_fall = False
    if in_fall:
        ax.axvspan(t_start, df[time_col].iloc[-1], color=color,
                   alpha=FALL_ALPHA)


def plot_accelerometer(df_orig, df_sim, save_dir: Path):
    fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex="col")
    fig.suptitle("Accelerometer Comparison  -  Original vs Simulated\n"
                 "Task 34: Backward fall while walking (slip)",
                 fontsize=13, fontweight="bold", y=0.98)

    axes_orig = axes[:, 0]
    axes_sim  = axes[:, 1]

    channels = [("AccX_ms2", "Acc X (m/s²)"),
                ("AccY_ms2", "Acc Y (m/s²)"),
                ("AccZ_ms2", "Acc Z (m/s²)"),
                ("accel_mag_ms2", "Acc Magnitude (m/s²)")]

    for i, (col, ylabel) in enumerate(channels):
        # -- Original ----------------------------------------------------
        ax = axes_orig[i]
        ax.plot(df_orig["time_s"], df_orig[col],
                color=ORIG_COLOR, linewidth=0.8, label="Original S23")
        _shade_fall(ax, df_orig, color="red")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(f"Original - {col}", fontsize=9)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

        # -- Simulated ---------------------------------------------------
        ax = axes_sim[i]
        ax.plot(df_sim["time_s"], df_sim[col],
                color=SIM_COLOR, linewidth=0.8, label="Simulated")
        _shade_fall(ax, df_sim, color="red")
        ax.set_title(f"Simulated - {col}", fontsize=9)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)", fontsize=9)

    plt.tight_layout()
    out = save_dir / "plot_accelerometer.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[PLOT] Accelerometer ? {out}")
    plt.show()


def plot_gyroscope(df_orig, df_sim, save_dir: Path):
    fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex="col")
    fig.suptitle("Gyroscope Comparison  -  Original vs Simulated\n"
                 "Task 34: Backward fall while walking (slip)",
                 fontsize=13, fontweight="bold", y=0.98)

    axes_orig = axes[:, 0]
    axes_sim  = axes[:, 1]

    channels = [("GyrX_rads", "Gyr X (rad/s)"),
                ("GyrY_rads", "Gyr Y (rad/s)"),
                ("GyrZ_rads", "Gyr Z (rad/s)"),
                ("gyro_mag_dps",  "Gyr Magnitude (dps)")]

    for i, (col, ylabel) in enumerate(channels):
        ax = axes_orig[i]
        ax.plot(df_orig["time_s"], df_orig[col],
                color=ORIG_COLOR, linewidth=0.8, label="Original S23")
        _shade_fall(ax, df_orig, color="red")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(f"Original - {col}", fontsize=9)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

        ax = axes_sim[i]
        ax.plot(df_sim["time_s"], df_sim[col],
                color=SIM_COLOR, linewidth=0.8, label="Simulated")
        _shade_fall(ax, df_sim, color="red")
        ax.set_title(f"Simulated - {col}", fontsize=9)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)", fontsize=9)

    plt.tight_layout()
    out = save_dir / "plot_gyroscope.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[PLOT] Gyroscope ? {out}")
    plt.show()


def plot_accel_magnitude_overlay(df_orig, df_sim, save_dir: Path):
    """
    Single-panel overlay: accel magnitude for both, time-normalised
    to the fall window so the shapes are directly comparable.
    """
    # -- extract fall window for each ------------------------------------
    def window(df, pad_s=3.0):
        fall_rows = df[df["fall_flag"] == 1]
        if fall_rows.empty:
            return df, df["time_s"].iloc[0]
        t_fall = fall_rows["time_s"].iloc[0]
        t_start = max(df["time_s"].iloc[0], t_fall - pad_s)
        t_end   = min(df["time_s"].iloc[-1], t_fall + pad_s * 2)
        win = df[(df["time_s"] >= t_start) & (df["time_s"] <= t_end)].copy()
        win["t_rel"] = win["time_s"] - t_fall
        return win, t_fall

    win_orig, _ = window(df_orig)
    win_sim,  _ = window(df_sim)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("Acceleration Magnitude - Fall Window Overlay\n"
                 "t=0 aligned to fall onset",
                 fontsize=12, fontweight="bold")

    # raw overlay
    ax = axes[0]
    ax.plot(win_orig["t_rel"], win_orig["accel_mag_ms2"],
            color=ORIG_COLOR, linewidth=1.2, label="Original S23")
    ax.plot(win_sim["t_rel"],  win_sim["accel_mag_ms2"],
            color=SIM_COLOR,  linewidth=1.2, label="Simulated", linestyle="--")
    ax.axvline(0, color="red", linestyle=":", linewidth=1.5, label="fall onset")
    ax.set_ylabel("|Acc| (m/s²)", fontsize=10)
    ax.set_title("Acceleration Magnitude", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # normalised (0-1) overlay
    ax = axes[1]
    def norm01(s):
        r = s.max() - s.min()
        return (s - s.min()) / r if r > 0 else s * 0

    ax.plot(win_orig["t_rel"], norm01(win_orig["accel_mag_ms2"]),
            color=ORIG_COLOR, linewidth=1.2, label="Original S23 (norm)")
    ax.plot(win_sim["t_rel"],  norm01(win_sim["accel_mag_ms2"]),
            color=SIM_COLOR,  linewidth=1.2, label="Simulated (norm)", linestyle="--")
    ax.axvline(0, color="red", linestyle=":", linewidth=1.5, label="fall onset")
    ax.set_ylabel("Normalised |Acc|", fontsize=10)
    ax.set_xlabel("Time relative to fall onset (s)", fontsize=10)
    ax.set_title("Normalised Acceleration Magnitude (shape comparison)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = save_dir / "plot_accel_overlay.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[PLOT] Accel overlay ? {out}")
    plt.show()


def plot_gyro_magnitude_overlay(df_orig, df_sim, save_dir: Path):
    """Same overlay idea for gyro magnitude."""
    def window(df, pad_s=3.0):
        fall_rows = df[df["fall_flag"] == 1]
        if fall_rows.empty:
            return df
        t_fall = fall_rows["time_s"].iloc[0]
        t_start = max(df["time_s"].iloc[0], t_fall - pad_s)
        t_end   = min(df["time_s"].iloc[-1], t_fall + pad_s * 2)
        win = df[(df["time_s"] >= t_start) & (df["time_s"] <= t_end)].copy()
        win["t_rel"] = win["time_s"] - t_fall
        return win

    win_orig = window(df_orig)
    win_sim  = window(df_sim)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("Gyroscope Magnitude - Fall Window Overlay\n"
                 "t=0 aligned to fall onset",
                 fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.plot(win_orig["t_rel"], win_orig["gyro_mag_dps"],
            color=ORIG_COLOR, linewidth=1.2, label="Original S23")
    ax.plot(win_sim["t_rel"],  win_sim["gyro_mag_dps"],
            color=SIM_COLOR,  linewidth=1.2, label="Simulated", linestyle="--")
    ax.axvline(0, color="red", linestyle=":", linewidth=1.5, label="fall onset")
    ax.set_ylabel("|Gyr| (dps)", fontsize=10)
    ax.set_title("Gyroscope Magnitude", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    def norm01(s):
        r = s.max() - s.min()
        return (s - s.min()) / r if r > 0 else s * 0

    ax = axes[1]
    ax.plot(win_orig["t_rel"], norm01(win_orig["gyro_mag_dps"]),
            color=ORIG_COLOR, linewidth=1.2, label="Original S23 (norm)")
    ax.plot(win_sim["t_rel"],  norm01(win_sim["gyro_mag_dps"]),
            color=SIM_COLOR,  linewidth=1.2, label="Simulated (norm)", linestyle="--")
    ax.axvline(0, color="red", linestyle=":", linewidth=1.5, label="fall onset")
    ax.set_ylabel("Normalised |Gyr|", fontsize=10)
    ax.set_xlabel("Time relative to fall onset (s)", fontsize=10)
    ax.set_title("Normalised Gyroscope Magnitude (shape comparison)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = save_dir / "plot_gyro_overlay.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[PLOT] Gyro overlay ? {out}")
    plt.show()


def plot_statistics_summary(df_orig, df_sim, save_dir: Path):
    """
    Bar chart comparison of key statistics for the fall window only.
    """
    def fall_stats(df):
        fall = df[df["fall_flag"] == 1]
        pre  = df[df["fall_flag"] == 0]
        return {
            "Peak |Acc| (m/s²)": fall["accel_mag_ms2"].max(),
            "Mean |Acc| pre-fall": pre["accel_mag_ms2"].mean(),
            "Mean |Acc| during fall": fall["accel_mag_ms2"].mean(),
            "Peak |Gyr| (dps)": fall["gyro_mag_dps"].max(),
            "Mean |Gyr| during fall": fall["gyro_mag_dps"].mean(),
        }

    stats_orig = fall_stats(df_orig)
    stats_sim  = fall_stats(df_sim)

    metrics = list(stats_orig.keys())
    vals_orig = [stats_orig[m] for m in metrics]
    vals_sim  = [stats_sim[m]  for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, vals_orig, width, label="Original S23",
                   color=ORIG_COLOR, alpha=0.85)
    bars2 = ax.bar(x + width/2, vals_sim,  width, label="Simulated",
                   color=SIM_COLOR,  alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Value", fontsize=10)
    ax.set_title("Statistical Summary - Fall Window\n"
                 "Task 34: Backward fall while walking (slip)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    out = save_dir / "plot_statistics_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[PLOT] Statistics summary ? {out}")
    plt.show()

    # -- print numeric table ----------------------------------------------
    print("\n-- Statistical Comparison Table ----------------------------------")
    print(f"{'Metric':<35} {'Original':>12} {'Simulated':>12} {'Ratio S/O':>10}")
    print("-" * 72)
    for m in metrics:
        o = stats_orig[m]
        s = stats_sim[m]
        r = s / o if o != 0 else float("nan")
        print(f"{m:<35} {o:>12.3f} {s:>12.3f} {r:>10.3f}")
    print("-" * 72)


# -----------------------------------------------------------
#  MAIN
# -----------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("  IMU Fall Data Analysis")
    print("=" * 60)

    # 1. Load & reshape
    df_sim  = load_simulated(SIM_RAW)
    df_orig, onset_frame, impact_frame = load_original(
        ORIG_FILE, LABEL_FILE, task_id=TASK_ID, trial_id=TRIAL_ID)

    # 2. Save cleaned simulated CSV
    save_cleaned_sim(df_sim, OUTPUT_DIR)

    # 3. Plots
    print("\nGenerating plots ...")
    plot_accelerometer(df_orig, df_sim, OUTPUT_DIR)
    plot_gyroscope(df_orig, df_sim, OUTPUT_DIR)
    plot_accel_magnitude_overlay(df_orig, df_sim, OUTPUT_DIR)
    plot_gyro_magnitude_overlay(df_orig, df_sim, OUTPUT_DIR)
    plot_statistics_summary(df_orig, df_sim, OUTPUT_DIR)

    print("\n[DONE] All outputs saved to:", OUTPUT_DIR)