#!/usr/bin/env python3
"""
Scientific Visualization for Entropy and Logit Telemetry Metrics.
Generates publication-quality plots from metrics_details.csv.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Apply Global Publication Typography and Style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Inter', 'Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'figure.autolayout': False,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLORS = {
    "B1": "#6C757D",  # Slate Gray (Pure Lexical)
    "B2": "#2B5C8F",  # Cool Blue (Pure Dense)
    "B3": "#2A9D8F",  # Emerald Green (Hybrid + Rerank)
    "B4": "#E76F51",  # Warm Coral (Graph + Rerank)
    "B5": "#E63946",  # Crimson Red (Full Pipeline)
}

def generate_boxplots_baseline(df, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [("citation_entropy", "Citation Entropy (bits)"), 
               ("clr", "CLR (Confidence Log Ratio)"), 
               ("ll_rag", "LL RAG (Log Likelihood)")]
    
    target_order = ["B1", "B2", "B3", "B4", "B5"]
    baselines = [b for b in target_order if b in df['baseline'].unique()]
    
    for ax, (col, title) in zip(axes, metrics):
        if col in df.columns:
            sns.boxplot(data=df, x="baseline", y=col, ax=ax, order=baselines, palette=COLORS, showfliers=False)
            sns.stripplot(data=df, x="baseline", y=col, ax=ax, order=baselines, color='black', alpha=0.3, size=2, jitter=True)
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel("Baseline")
            ax.set_ylabel(title)
            ax.grid(True, linestyle="--", alpha=0.6, color="#EAEAEA")
            sns.despine(ax=ax)
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_entropy_by_baseline.png"))
    plt.savefig(os.path.join(output_dir, "01_entropy_by_baseline.pdf"))
    plt.close()

def generate_scatter_quality(df, output_dir):
    if "citation_entropy" not in df.columns or "semantic_accuracy" not in df.columns:
        return
        
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    target_order = ["B1", "B2", "B3", "B4", "B5"]
    baselines = [b for b in target_order if b in df['baseline'].unique()]
    
    # Filter out missing values for regression
    df_clean = df.dropna(subset=["citation_entropy", "semantic_accuracy", "clr"])
    
    sns.scatterplot(data=df, x="citation_entropy", y="semantic_accuracy", hue="baseline", 
                    hue_order=baselines, palette=COLORS, alpha=0.6, ax=axes[0])
    sns.regplot(data=df_clean, x="citation_entropy", y="semantic_accuracy", scatter=False, 
                color='black', ax=axes[0], line_kws={"linestyle":"--", "alpha":0.5})
    axes[0].set_title("Citation Entropy vs Semantic Accuracy", fontweight='bold')
    axes[0].set_xlabel("Citation Entropy (bits)")
    axes[0].set_ylabel("Semantic Accuracy")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    sns.despine(ax=axes[0])
    
    if "clr" in df.columns:
        sns.scatterplot(data=df, x="clr", y="semantic_accuracy", hue="baseline", 
                        hue_order=baselines, palette=COLORS, alpha=0.6, ax=axes[1])
        sns.regplot(data=df_clean, x="clr", y="semantic_accuracy", scatter=False, 
                    color='black', ax=axes[1], line_kws={"linestyle":"--", "alpha":0.5})
        axes[1].set_title("CLR vs Semantic Accuracy", fontweight='bold')
        axes[1].set_xlabel("CLR (Confidence Log Ratio)")
        axes[1].set_ylabel("Semantic Accuracy")
        axes[1].grid(True, linestyle="--", alpha=0.6)
        sns.despine(ax=axes[1])
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_entropy_quality_scatter.png"))
    plt.savefig(os.path.join(output_dir, "02_entropy_quality_scatter.pdf"))
    plt.close()

def generate_correlation_heatmap(df, output_dir):
    entropy_cols = ["citation_entropy", "ll_rag", "ll_base", "clr", "msp", "logit_margin"]
    quality_cols = ["semantic_accuracy", "faithfulness", "answer_relevance", "retrieval_recall", "context_precision"]
    
    available_cols = [c for c in entropy_cols + quality_cols if c in df.columns]
    
    if len(available_cols) < 2:
        return
        
    # Convert to numeric, errors='coerce' to turn unparseable into NaN
    df_numeric = df[available_cols].apply(pd.to_numeric, errors='coerce')
    corr = df_numeric.corr(method='pearson')
    
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="vlag", vmin=-1.0, vmax=1.0, center=0,
                annot=True, fmt=".2f", square=True, linewidths=.8,
                cbar_kws={"shrink": .8, "label": r"Pearson Correlation Coefficient ($r$)"})
    plt.title("Correlation: Entropy vs. Quality Metrics", fontweight='bold', pad=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_entropy_correlation_heatmap.png"))
    plt.savefig(os.path.join(output_dir, "03_entropy_correlation_heatmap.pdf"))
    plt.close()

def generate_boxplots_answerability(df, output_dir):
    if "answerability_outcome" not in df.columns or "citation_entropy" not in df.columns:
        return
        
    fig, ax = plt.subplots(figsize=(8, 6))
    
    order = ["TP", "TN", "FP", "FN"]
    order = [o for o in order if o in df["answerability_outcome"].unique()]
    
    outcome_colors = {"TP": "#2A9D8F", "TN": "#2B5C8F", "FP": "#E63946", "FN": "#F4A261"}
    
    sns.boxplot(data=df, x="answerability_outcome", y="citation_entropy", order=order, 
                palette=outcome_colors, ax=ax, showfliers=False)
    sns.stripplot(data=df, x="answerability_outcome", y="citation_entropy", order=order,
                  color='black', alpha=0.3, size=3, ax=ax, jitter=True)
                  
    ax.set_title("Citation Entropy by Answerability Outcome", fontweight='bold')
    ax.set_xlabel("Answerability Outcome (TP/TN = Correct, FP/FN = Error)")
    ax.set_ylabel("Citation Entropy (bits)")
    ax.grid(True, linestyle="--", alpha=0.6, axis='y')
    sns.despine(ax=ax)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_entropy_by_answerability.png"))
    plt.savefig(os.path.join(output_dir, "04_entropy_by_answerability.pdf"))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate entropy visualizations")
    parser.add_argument("--input", required=True, help="Path to run directory")
    args = parser.parse_args()
    
    details_csv = os.path.join(args.input, "metrics_details.csv")
    if not os.path.exists(details_csv):
        print(f"Error: Could not find {details_csv}")
        return
        
    output_dir = os.path.join(args.input, "figures_entropy")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {details_csv}...")
    try:
        df = pd.read_csv(details_csv)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        # Try reading with different parameters if there are quoting issues
        df = pd.read_csv(details_csv, on_bad_lines='skip')
        print(f"Loaded with skipped lines. Shape: {df.shape}")
    
    # Exclude legacy B3 and map legacy B4-B6 -> B3-B5 for presentation
    if "baseline" in df.columns:
        unique_b = set(df['baseline'].dropna().unique())
        is_legacy = ("B6" in unique_b) or ("B3" not in unique_b and any(b in unique_b for b in ["B4", "B5"]))
        if is_legacy:
            legacy_remap = {"B1": "B1", "B2": "B2", "B4": "B3", "B5": "B4", "B6": "B5"}
            df = df[df['baseline'] != 'B3'].copy()
            df['baseline'] = df['baseline'].map(legacy_remap)
            df = df.dropna(subset=['baseline'])
        else:
            df = df[df['baseline'].isin(["B1", "B2", "B3", "B4", "B5"])].copy()

    print("Generating Boxplots by Baseline...")
    generate_boxplots_baseline(df, output_dir)
    
    print("Generating Scatter Plots (Entropy vs Quality)...")
    generate_scatter_quality(df, output_dir)
    
    print("Generating Correlation Heatmap...")
    generate_correlation_heatmap(df, output_dir)
    
    print("Generating Boxplots by Answerability Outcome...")
    generate_boxplots_answerability(df, output_dir)
    
    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()
