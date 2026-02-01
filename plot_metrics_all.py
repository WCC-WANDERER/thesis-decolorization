import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

OUTPUT_DIR = "plots"
CSV_PATTERN = "metrics_*.csv"  # match all metrics CSV files
METHOD_NAME_MAP = {
    "average": "Average",
    "luminance": "Luminance",
    "cielab": "CIELAB",
    "decolorize": "GD",
    "corrc2g": "CorrC2G",
    "color2gray": "Color2Gray"
}

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
    means = df.groupby("Method_display")[metric_name].mean().sort_values(ascending=False)
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
    df_all["Method_display"] = df_all["Method"].map(METHOD_NAME_MAP)

    # Map DataFrame columns to correct titles
    metrics_map = {
        "Time(sec)": "Time(sec)",
        "RMS": "RMS",
        "NRMS": "NRMS",
        "GRR": "GRR",
        "C2G-SSIM": "C2G-SSIM",
        "CCPR": "CCPR",
        "CCFR": "CCFR",
        "E-Score": "E-Score"
    }

    for col, title in metrics_map.items():
        plot_bar(df_all, col, title=title)

    print(f"All plots saved to '{OUTPUT_DIR}' folder. Methods compared: {df_all['Method_display'].unique()}")

if __name__ == "__main__":
    main()
