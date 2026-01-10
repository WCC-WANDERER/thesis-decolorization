import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

OUTPUT_DIR = "plots"
CSV_PATTERN = "metrics_*.csv"  # match all metrics CSV files

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_all_csv(pattern):
    """
    Load all CSV files matching pattern and concatenate into a single DataFrame
    """
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No CSV files found with pattern {pattern}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    return combined

def plot_bar(df, metric_name, title=None):
    """
    Plot average metric per method as bar chart
    """
    plt.figure(figsize=(10, 5))
    means = df.groupby("method")[metric_name].mean().sort_values(ascending=False)
    means.plot(kind="bar", color='skyblue')
    plt.title(title if title else metric_name)
    plt.ylabel(metric_name)
    plt.xlabel("Method")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{metric_name}.png", dpi=200)
    plt.close()

def main():
    ensure_dir(OUTPUT_DIR)
    df_all = load_all_csv(CSV_PATTERN)

    # Map DataFrame columns to pretty titles
    metrics_map = {
        "time_sec": "Time(sec)",
        "rms": "RMS",
        "nrms": "NRMS",
        "c2g_ssim": "C2G_SSIM",
        "fsim": "FSIM",
        "grr": "GRR",
        "ccpr": "CCPR",
        "ccfr": "CCFR",
        "escore": "E-Score"
    }

    for col, title in metrics_map.items():
        plot_bar(df_all, col, title=title)

    print(f"All plots saved to '{OUTPUT_DIR}' folder. Methods compared: {df_all['method'].unique()}")

if __name__ == "__main__":
    main()
