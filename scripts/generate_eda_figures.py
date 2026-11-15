"""Script to generate all EDA figures strictly on the development set (N=5,634)."""

import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from telco_churn.data import load_raw_data
from telco_churn.schema import validate_records

# Configure seaborn & matplotlib
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["font.size"] = 11

OUTPUT_DIR = "reports/figures/eda"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHURN_PALETTE = {0: "#2b5c8f", 1: "#d95f02", "0": "#2b5c8f", "1": "#d95f02"}


def get_dev_data():
    raw_path = "telco-customer-churn-on-icp4d/data/Telco-Customer-Churn.csv"
    manifest_path = "data/splits/split_manifest.csv"

    df_raw = load_raw_data(raw_path)
    valid_df, _ = validate_records(df_raw, mode="training")
    manifest_df = pd.read_csv(manifest_path)

    dev_manifest = manifest_df[manifest_df["split"] == "dev"]
    dev_ids = set(dev_manifest["customerID"])
    dev_df = valid_df[valid_df["customerID"].isin(dev_ids)].copy()
    dev_df["Churn"] = dev_df["Churn"].astype(int)
    return dev_df


def plot_01_target_imbalance(dev_df):
    plt.figure(figsize=(7, 5))
    counts = dev_df["Churn"].value_counts().sort_index()
    labels = ["Khong roi bo (No: 0)", "Roi bo (Yes: 1)"]
    colors = ["#2b5c8f", "#d95f02"]
    
    bars = plt.bar(labels, counts.values, color=colors, width=0.55, edgecolor="black", alpha=0.85)
    total = len(dev_df)
    
    for bar in bars:
        height = bar.get_height()
        pct = (height / total) * 100
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 50,
            f"{height:,} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.title(f"Phan bo nhan muc tieu Churn tren tap phat trien (N = {total:,})", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Trang thai Churn", fontsize=11, labelpad=10)
    plt.ylabel("So luong mau (khach hang)", fontsize=11, labelpad=10)
    plt.ylim(0, max(counts.values) * 1.15)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "01_target_imbalance.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {out_path}")


def plot_02_tenure(dev_df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    total = len(dev_df)

    # Histogram & KDE
    sns.histplot(
        data=dev_df,
        x="tenure",
        hue="Churn",
        kde=True,
        bins=36,
        palette=CHURN_PALETTE,
        ax=ax1,
        element="step",
        common_norm=False,
    )
    ax1.set_title("Phan bo tham nien (tenure) theo Churn", fontsize=12, fontweight="bold")
    ax1.set_xlabel("So thang su dung (tenure)", fontsize=10)
    ax1.set_ylabel("Mat do / So luong", fontsize=10)
    if ax1.get_legend():
        ax1.legend(["Roi bo (1)", "Khong roi bo (0)"], title="Churn")

    # Boxplot
    sns.boxplot(
        data=dev_df,
        x="Churn",
        y="tenure",
        hue="Churn",
        legend=False,
        palette=CHURN_PALETTE,
        ax=ax2,
        width=0.45,
    )
    ax2.set_title("Boxplot tham nien (tenure) theo Churn", fontsize=12, fontweight="bold")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Khong roi bo (0)", "Roi bo (1)"])
    ax2.set_xlabel("Trang thai Churn", fontsize=10)
    ax2.set_ylabel("So thang su dung (tenure)", fontsize=10)

    fig.suptitle(f"Quan he giua tham nien va nguy co roi bo (Tap phat trien N = {total:,})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "02_tenure_distribution.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {out_path}")


def plot_03_contract(dev_df):
    fig, ax1 = plt.subplots(figsize=(9, 5))
    total = len(dev_df)

    contract_stats = dev_df.groupby("Contract")["Churn"].agg(
        total_count="count",
        churn_count="sum",
        churn_rate="mean"
    ).reindex(["Month-to-month", "One year", "Two year"])

    x = np.arange(len(contract_stats))
    width = 0.4

    # Bar chart for counts
    ax1.bar(x, contract_stats["total_count"], width, label="Tong so khach", color="#6baed6", edgecolor="black", alpha=0.8)
    ax1.bar(x, contract_stats["churn_count"], width, label="Khach roi bo", color="#d95f02", edgecolor="black", alpha=0.9)
    
    ax1.set_ylabel("So luong mau (khach hang)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(contract_stats.index, fontsize=11)
    ax1.set_xlabel("Loai hop dong (Contract)", fontsize=11, labelpad=10)

    # Line plot for churn rate
    ax2 = ax1.twinx()
    ax2.plot(x, contract_stats["churn_rate"] * 100, color="#b30000", marker="o", linewidth=2.5, markersize=8, label="Ty le roi bo (%)")
    ax2.set_ylabel("Ty le roi bo (%)", color="#b30000", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#b30000")
    ax2.set_ylim(0, 60)

    for i, rate in enumerate(contract_stats["churn_rate"]):
        ax2.text(i, rate * 100 + 2.5, f"{rate*100:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold", color="#b30000")

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title(f"Ty le roi bo va quy mo mau theo hop dong (Tap phat trien N = {total:,})", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "03_contract_churn.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {out_path}")


def plot_04_charges(dev_df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    total = len(dev_df)

    # MonthlyCharges KDE
    sns.kdeplot(
        data=dev_df,
        x="MonthlyCharges",
        hue="Churn",
        common_norm=False,
        palette=CHURN_PALETTE,
        fill=True,
        alpha=0.3,
        ax=ax1,
    )
    ax1.set_title("Mat do phi hang thang (MonthlyCharges)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Cuoc hang thang (USD)", fontsize=10)
    ax1.set_ylabel("Mat do phan bo (Density)", fontsize=10)
    if ax1.get_legend():
        ax1.legend(["Roi bo (1)", "Khong roi bo (0)"], title="Churn")

    # TotalCharges Boxplot
    sns.boxplot(
        data=dev_df,
        x="Churn",
        y="TotalCharges",
        hue="Churn",
        legend=False,
        palette=CHURN_PALETTE,
        ax=ax2,
        width=0.45,
    )
    ax2.set_title("Boxplot tong chi phi tich luy (TotalCharges)", fontsize=12, fontweight="bold")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Khong roi bo (0)", "Roi bo (1)"])
    ax2.set_xlabel("Trang thai Churn", fontsize=10)
    ax2.set_ylabel("Tong cuoc luy ke (USD)", fontsize=10)

    fig.suptitle(f"So sanh chi phi cuoc giua hai nhom khach hang (Tap phat trien N = {total:,})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "04_charges_distribution.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {out_path}")


def plot_05_services(dev_df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    total = len(dev_df)

    cols = ["InternetService", "TechSupport", "PaymentMethod"]
    titles = ["Dich vu Internet", "Ho tro ky thuat", "Phuong thuc thanh toan"]

    for ax, col, title in zip(axes, cols, titles):
        stats = dev_df.groupby(col)["Churn"].agg(
            total="count",
            rate="mean"
        ).sort_values(by="rate", ascending=False)

        bars = ax.bar(
            range(len(stats)),
            stats["rate"] * 100,
            color="#d95f02",
            edgecolor="black",
            alpha=0.8,
            width=0.5,
        )
        ax.set_xticks(range(len(stats)))
        ax.set_xticklabels(stats.index, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Ty le Churn (%)", fontsize=10)
        ax.set_ylim(0, 60)
        ax.set_title(title, fontsize=11, fontweight="bold")

        for i, bar in enumerate(bars):
            height = bar.get_height()
            cnt = stats["total"].iloc[i]
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1.5,
                f"{height:.1f}%\n(N={cnt})",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle(f"Ty le roi bo theo Dich vu, Ho tro va Thanh toan (Tap phat trien N = {total:,})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "05_service_payment_churn.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {out_path}")


def plot_06_correlation(dev_df):
    plt.figure(figsize=(7, 5))
    total = len(dev_df)

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    corr = dev_df[num_cols].corr()

    sns.heatmap(
        corr,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=1,
        cbar_kws={"label": "He so tuong quan Pearson"},
    )
    plt.title(f"Ma tran tuong quan giua cac bien so (Tap phat trien N = {total:,})", fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "06_numeric_correlation.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {out_path}")


def main():
    print("[*] Generating EDA figures strictly on the development set...")
    dev_df = get_dev_data()
    print(f"[*] Loaded dev_df with {len(dev_df):,} records.")

    plot_01_target_imbalance(dev_df)
    plot_02_tenure(dev_df)
    plot_03_contract(dev_df)
    plot_04_charges(dev_df)
    plot_05_services(dev_df)
    plot_06_correlation(dev_df)
    print("[+] All EDA figures successfully generated!")


if __name__ == "__main__":
    main()
