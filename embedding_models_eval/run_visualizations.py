"""
Visualizacoes de variabilidade por prompt: boxplots e bootstrap CI.

Carrega os CSVs de metricas por prompt (gerados por run_ranking_only e
run_tfidf_baseline) e produz:

1. Boxplots comparativos entre modelos para metricas-chave (F1@5, MAP@5, Norm@5)
2. Bootstrap confidence intervals (95%) para cada modelo/metrica
3. Tabela CSV consolidada com media, desvio padrao e intervalo de confianca

Execucao apartada do pipeline principal -- nao interfere nas entregas.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Configuracao visual
# ---------------------------------------------------------------------------

FIGSIZE_BOXPLOT: Tuple[int, int] = (14, 6)
FIGSIZE_VIOLIN: Tuple[int, int] = (14, 6)
DPI: int = 200
PALETTE: str = "Set2"
FONT_SIZE: int = 11

METRICS_OF_INTEREST: Dict[str, str] = {
    "F1@5": "F1@5",
    "AP@5": "MAP@5 (per-prompt AP)",
    "norm_mean_votes@5": "Norm@5",
}

MODEL_DISPLAY_ORDER: List[str] = [
    "tfidf",
    "minilm",
    "minilm_l12",
    "mpnet_base",
    "multilingual",
    "paraphrase_minilm",
    "multi_qa_minilm",
    "openai_small",
    "openai_large",
]

MODEL_LABELS: Dict[str, str] = {
    "tfidf": "TF-IDF",
    "minilm": "MiniLM-L6",
    "minilm_l12": "MiniLM-L12",
    "mpnet_base": "MPNet",
    "multilingual": "Multilingual",
    "paraphrase_minilm": "Paraphrase",
    "multi_qa_minilm": "MultiQA",
    "openai_small": "OpenAI-S",
    "openai_large": "OpenAI-L",
}


def _extract_model_name(filepath: Path) -> str:
    """
    Extrai nome do modelo a partir do caminho do CSV.

    Convencao:
        ranking_only/<modelo>_metrics_per_prompt.csv
        tfidf_baseline/tfidf_metrics_per_prompt.csv
    """
    stem = filepath.stem
    suffix = "_metrics_per_prompt"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def load_all_per_prompt(
    ranking_dir: str = "results/ranking_only",
    tfidf_dir: str = "results/tfidf_baseline",
) -> pd.DataFrame:
    """
    Carrega e concatena todos os CSVs de metricas por prompt.

    Args:
        ranking_dir: Diretorio com CSVs dos modelos de embeddings.
        tfidf_dir: Diretorio com CSV do baseline TF-IDF.

    Returns:
        DataFrame com colunas originais + 'model'.

    Raises:
        FileNotFoundError: Se nenhum CSV for encontrado.
    """
    frames: List[pd.DataFrame] = []

    for directory in [ranking_dir, tfidf_dir]:
        dir_path = Path(directory)
        if not dir_path.exists():
            continue
        for csv_path in sorted(dir_path.glob("*_metrics_per_prompt.csv")):
            model_name = _extract_model_name(csv_path)
            df = pd.read_csv(csv_path)
            df["model"] = model_name
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"Nenhum CSV encontrado em {ranking_dir} ou {tfidf_dir}"
        )

    combined = pd.concat(frames, ignore_index=True)
    print(f"   Carregados {len(frames)} modelos, {len(combined)} linhas total")
    return combined


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Calcula intervalo de confianca via bootstrap.

    Args:
        values: Array de valores observados (e.g. 20 prompts).
        n_bootstrap: Numero de reamostras.
        ci: Nivel de confianca (0-1).
        seed: Semente para reprodutibilidade.

    Returns:
        Tupla (media, limite_inferior, limite_superior).
    """
    rng = np.random.default_rng(seed)
    n = len(values)

    if n == 0:
        return (np.nan, np.nan, np.nan)

    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    alpha = 1.0 - ci
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1.0 - alpha / 2))
    mean = np.mean(values)

    return (mean, lower, upper)


def compute_bootstrap_table(
    df: pd.DataFrame,
    metrics: Optional[Dict[str, str]] = None,
    n_bootstrap: int = 10000,
) -> pd.DataFrame:
    """
    Computa tabela de bootstrap CI para todos os modelos e metricas.

    Args:
        df: DataFrame com colunas de metricas + 'model'.
        metrics: Dicionario {coluna_csv: label_display}.
        n_bootstrap: Numero de reamostras bootstrap.

    Returns:
        DataFrame com colunas: model, metric, mean, std, ci_lower, ci_upper.
    """
    if metrics is None:
        metrics = METRICS_OF_INTEREST

    rows: List[Dict[str, object]] = []

    for model_name in df["model"].unique():
        model_df = df[df["model"] == model_name]
        for col, label in metrics.items():
            if col not in model_df.columns:
                continue
            values = model_df[col].dropna().values
            mean, ci_lo, ci_hi = bootstrap_ci(values, n_bootstrap=n_bootstrap)
            rows.append({
                "model": model_name,
                "model_label": MODEL_LABELS.get(model_name, model_name),
                "metric": label,
                "metric_col": col,
                "mean": mean,
                "std": np.std(values, ddof=1) if len(values) > 1 else 0.0,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "n_prompts": len(values),
            })

    return pd.DataFrame(rows)


def plot_boxplots(
    df: pd.DataFrame,
    metrics: Optional[Dict[str, str]] = None,
    output_dir: str = "results/visualizations",
) -> List[Path]:
    """
    Gera boxplots comparativos entre modelos para cada metrica.

    Args:
        df: DataFrame com colunas de metricas + 'model'.
        metrics: Dicionario {coluna_csv: label_display}.
        output_dir: Diretorio de saida.

    Returns:
        Lista de caminhos dos PNGs gerados.
    """
    if metrics is None:
        metrics = METRICS_OF_INTEREST

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    order = [m for m in MODEL_DISPLAY_ORDER if m in df["model"].unique()]
    labels = [MODEL_LABELS.get(m, m) for m in order]

    for col, display_name in metrics.items():
        if col not in df.columns:
            print(f"   AVISO: coluna '{col}' ausente, pulando")
            continue

        fig, ax = plt.subplots(figsize=FIGSIZE_BOXPLOT)

        sns.boxplot(
            data=df,
            x="model",
            y=col,
            hue="model",
            order=order,
            hue_order=order,
            palette=PALETTE,
            width=0.6,
            linewidth=1.2,
            fliersize=4,
            legend=False,
            ax=ax,
        )

        sns.stripplot(
            data=df,
            x="model",
            y=col,
            order=order,
            color="0.3",
            size=3,
            alpha=0.5,
            jitter=True,
            ax=ax,
        )

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=FONT_SIZE - 1)
        ax.set_ylabel(display_name, fontsize=FONT_SIZE)
        ax.set_xlabel("")
        ax.set_title(
            f"Distribution of {display_name} across 20 prompts",
            fontsize=FONT_SIZE + 1,
            pad=12,
        )
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        sns.despine(ax=ax)

        fig.tight_layout()
        fname = out_path / f"boxplot_{col.replace('@', '_at_')}.png"
        fig.savefig(fname, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        saved.append(fname)
        print(f"   Salvo: {fname}")

    return saved


def plot_violin(
    df: pd.DataFrame,
    metrics: Optional[Dict[str, str]] = None,
    output_dir: str = "results/visualizations",
) -> List[Path]:
    """
    Gera violin plots comparativos (alternativa aos boxplots).

    Args:
        df: DataFrame com colunas de metricas + 'model'.
        metrics: Dicionario {coluna_csv: label_display}.
        output_dir: Diretorio de saida.

    Returns:
        Lista de caminhos dos PNGs gerados.
    """
    if metrics is None:
        metrics = METRICS_OF_INTEREST

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    order = [m for m in MODEL_DISPLAY_ORDER if m in df["model"].unique()]
    labels = [MODEL_LABELS.get(m, m) for m in order]

    for col, display_name in metrics.items():
        if col not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=FIGSIZE_VIOLIN)

        sns.violinplot(
            data=df,
            x="model",
            y=col,
            hue="model",
            order=order,
            hue_order=order,
            palette=PALETTE,
            inner="box",
            linewidth=1.0,
            cut=0,
            legend=False,
            ax=ax,
        )

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=FONT_SIZE - 1)
        ax.set_ylabel(display_name, fontsize=FONT_SIZE)
        ax.set_xlabel("")
        ax.set_title(
            f"Distribution of {display_name} across 20 prompts",
            fontsize=FONT_SIZE + 1,
            pad=12,
        )
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        sns.despine(ax=ax)

        fig.tight_layout()
        fname = out_path / f"violin_{col.replace('@', '_at_')}.png"
        fig.savefig(fname, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        saved.append(fname)
        print(f"   Salvo: {fname}")

    return saved


def plot_bootstrap_ci(
    ci_table: pd.DataFrame,
    output_dir: str = "results/visualizations",
) -> List[Path]:
    """
    Gera grafico de forest plot com media e IC 95% por modelo.

    Args:
        ci_table: DataFrame retornado por compute_bootstrap_table.
        output_dir: Diretorio de saida.

    Returns:
        Lista de caminhos dos PNGs gerados.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    order = [m for m in MODEL_DISPLAY_ORDER if m in ci_table["model"].unique()]

    for metric_name in ci_table["metric"].unique():
        subset = ci_table[ci_table["metric"] == metric_name].copy()
        subset["model"] = pd.Categorical(subset["model"], categories=order, ordered=True)
        subset = subset.sort_values("model")

        fig, ax = plt.subplots(figsize=(10, 5))

        y_positions = np.arange(len(subset))
        colors = sns.color_palette(PALETTE, n_colors=len(subset))

        for i, (_, row) in enumerate(subset.iterrows()):
            ax.errorbar(
                x=row["mean"],
                y=y_positions[i],
                xerr=[[row["mean"] - row["ci_lower"]], [row["ci_upper"] - row["mean"]]],
                fmt="o",
                color=colors[i],
                markersize=8,
                capsize=5,
                capthick=1.5,
                linewidth=1.5,
            )
            ax.text(
                row["ci_upper"] + 0.005,
                y_positions[i],
                f"{row['mean']:.3f} [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]",
                va="center",
                fontsize=FONT_SIZE - 2,
            )

        labels_y = [
            MODEL_LABELS.get(m, m) for m in subset["model"]
        ]
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels_y, fontsize=FONT_SIZE)
        ax.set_xlabel(metric_name, fontsize=FONT_SIZE)
        ax.set_title(
            f"Bootstrap 95% CI -- {metric_name} (10k resamples, n=20 prompts)",
            fontsize=FONT_SIZE + 1,
            pad=12,
        )
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        ax.invert_yaxis()
        sns.despine(ax=ax, left=True)

        fig.tight_layout()
        metric_slug = metric_name.lower().replace(" ", "_").replace("@", "_at_")
        metric_slug = metric_slug.replace("(", "").replace(")", "").replace("-", "")
        fname = out_path / f"bootstrap_ci_{metric_slug}.png"
        fig.savefig(fname, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        saved.append(fname)
        print(f"   Salvo: {fname}")

    return saved


def plot_combined_panel(
    df: pd.DataFrame,
    metrics: Optional[Dict[str, str]] = None,
    output_dir: str = "results/visualizations",
) -> Optional[Path]:
    """
    Gera painel combinado com boxplots de todas as metricas lado a lado.

    Args:
        df: DataFrame com colunas de metricas + 'model'.
        metrics: Dicionario {coluna_csv: label_display}.
        output_dir: Diretorio de saida.

    Returns:
        Caminho do PNG gerado, ou None em caso de erro.
    """
    if metrics is None:
        metrics = METRICS_OF_INTEREST

    available = {k: v for k, v in metrics.items() if k in df.columns}
    if not available:
        return None

    n_metrics = len(available)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6), sharey=False)

    if n_metrics == 1:
        axes = [axes]

    order = [m for m in MODEL_DISPLAY_ORDER if m in df["model"].unique()]
    labels = [MODEL_LABELS.get(m, m) for m in order]

    for ax, (col, display_name) in zip(axes, available.items()):
        sns.boxplot(
            data=df,
            x="model",
            y=col,
            hue="model",
            order=order,
            hue_order=order,
            palette=PALETTE,
            width=0.6,
            linewidth=1.0,
            fliersize=3,
            legend=False,
            ax=ax,
        )
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=FONT_SIZE - 2)
        ax.set_ylabel(display_name, fontsize=FONT_SIZE)
        ax.set_xlabel("")
        ax.set_title(display_name, fontsize=FONT_SIZE, pad=8)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        sns.despine(ax=ax)

    fig.suptitle(
        "Per-prompt metric distributions (n=20 prompts per model)",
        fontsize=FONT_SIZE + 2,
        y=1.02,
    )
    fig.tight_layout()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fname = out_path / "panel_boxplots_combined.png"
    fig.savefig(fname, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"   Salvo: {fname}")
    return fname


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Gera boxplots, violin plots e bootstrap CI "
            "para metricas por prompt de todos os modelos."
        ),
    )
    parser.add_argument(
        "--ranking-dir",
        type=str,
        default="results/ranking_only",
        help="Diretorio com CSVs dos modelos de embeddings",
    )
    parser.add_argument(
        "--tfidf-dir",
        type=str,
        default="results/tfidf_baseline",
        help="Diretorio com CSV do baseline TF-IDF",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/visualizations",
        help="Diretorio de saida para figuras e tabelas",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Numero de reamostras bootstrap (default: 10000)",
    )
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("VISUALIZACOES: BOXPLOTS + BOOTSTRAP CI")
    print("=" * 70)
    print()

    # -- 1. Carregar dados --
    print("1. Carregando metricas por prompt...")
    df = load_all_per_prompt(
        ranking_dir=args.ranking_dir,
        tfidf_dir=args.tfidf_dir,
    )
    print()

    # -- 2. Boxplots --
    print("2. Gerando boxplots...")
    plot_boxplots(df, output_dir=args.output_dir)
    print()

    # -- 3. Violin plots --
    print("3. Gerando violin plots...")
    plot_violin(df, output_dir=args.output_dir)
    print()

    # -- 4. Painel combinado --
    print("4. Gerando painel combinado...")
    plot_combined_panel(df, output_dir=args.output_dir)
    print()

    # -- 5. Bootstrap CI --
    print(f"5. Calculando bootstrap CI (n={args.n_bootstrap:,} reamostras)...")
    ci_table = compute_bootstrap_table(
        df,
        n_bootstrap=args.n_bootstrap,
    )

    ci_path = Path(args.output_dir) / "bootstrap_ci_table.csv"
    ci_table.to_csv(ci_path, index=False)
    print(f"   Tabela CI salva: {ci_path}")
    print()

    # Exibe resumo no terminal
    print("   Resumo Bootstrap 95% CI:")
    print("   " + "-" * 66)
    for _, row in ci_table.iterrows():
        label = row["model_label"]
        metric = row["metric"]
        mean = row["mean"]
        lo = row["ci_lower"]
        hi = row["ci_upper"]
        print(f"   {label:>15s} | {metric:<25s} | {mean:.4f} [{lo:.4f}, {hi:.4f}]")
    print()

    # -- 6. Forest plots --
    print("6. Gerando forest plots (CI)...")
    plot_bootstrap_ci(ci_table, output_dir=args.output_dir)
    print()

    print("=" * 70)
    print("CONCLUIDO")
    print("=" * 70)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
