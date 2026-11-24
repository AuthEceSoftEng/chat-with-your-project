"""
Self-contained asset generator for the "Chat with Your Project" paper
(chat-with-your-project-paper/paper.tex).

Reads the raw per-repo answer CSVs directly from ../outputs_{config}_{model}/
(relative to this script, i.e. tests/outputs/), using the three models and
four configurations of the current study:

  Models (weak -> medium -> strong): qwen2.5:7b, gpt-oss:20b, gpt-oss:120b
  Configs: Baseline (bf), Mongo-only, Full, Full+KG (kg)

Produces, all under this folder (tables/ and figures/):
  - table1_overall_effectiveness.csv / .tex   (Answer rate, Accuracy, Precision, repo-SD, Wilson CIs)
  - table2_tier_breakdown.csv / .tex          (Accuracy/Precision per tier)
  - table3_efficiency.csv / .tex              (tokens/tool-calls cost, incl. Full+KG)
  - table4_per_repository.csv / .tex          (per-repo Strict Accuracy, Mongo-only vs Full+KG)
  - figures/fig_overall_effectiveness.pdf     (stacked bar, CORRECT/INCORRECT/NO/ERROR)
  - figures/fig_tier_scaling.pdf              (per-tier grouped bar)
  - figures/fig_efficiency_frontier.pdf       (cost-accuracy scatter + token CDF)

This script is fully self-contained: it does not import or depend on
tests/outputs/evaluation/generate_evaluation.py.

Run from this folder:
    python generate_paper_assets.py
"""
import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.dirname(BASE)  # tests/outputs/
TABLE_DIR = os.path.join(BASE, "tables")
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'legend.title_fontsize': 12,
    'figure.titlesize': 12,
})

# ---------------------------------------------------------------------------
# Static metadata
# ---------------------------------------------------------------------------
REPO_FILES = {
    'apache_kafka_answers.csv': 'Kafka',
    'apache_tomcat_answers.csv': 'Tomcat',
    'hibernate_hibernate-orm_answers.csv': 'Hibernate',
    'jenkinsci_github-plugin_answers.csv': 'Jenkins',
    'spring-projects_spring-boot_answers.csv': 'SpringBoot',
}
REPOS = list(REPO_FILES.values())

CONFIGS = [
    ('bf', 'Baseline'),
    ('mongo_only', 'Mongo-only'),
    ('full', 'Full'),
    ('kg', 'KG'),
]

# Three models, in the exact order used throughout the paper:
# weak/smaller -> medium -> strong/larger
MODELS = [
    ('gpt20', 'gpt-oss:20b'),
    ('gpt', 'gpt-oss:120b'),
]

TIER_SLICES = {
    'Informational': (0, 20),
    'Computational': (20, 40),
    'Machine Learning': (40, 50),
}

DEFAULT_CYCLE = plt.rcParams['axes.prop_cycle'].by_key()['color']
DEFAULT_CYCLE = [DEFAULT_CYCLE[0], DEFAULT_CYCLE[3], DEFAULT_CYCLE[1], DEFAULT_CYCLE[7]]
STATUS_COLORS = {
    'Correct': DEFAULT_CYCLE[0],       # C0 blue
    'Incorrect': DEFAULT_CYCLE[1],     # C3 red
    'Refusal (NO)': DEFAULT_CYCLE[2],  # C1 orange
    'System Error': DEFAULT_CYCLE[3],  # C7 gray
}
MODEL_COLORS = {m[1]: DEFAULT_CYCLE[i] for i, m in enumerate(MODELS)}
CONFIG_COLORS = {c[1]: DEFAULT_CYCLE[i] for i, c in enumerate(CONFIGS)}
CONFIG_MARKERS = {'Baseline': 'o', 'Mongo-only': 's', 'Full': '^', 'KG': 'D'}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_all():
    data = {}
    for cfg_key, _ in CONFIGS:
        for model_key, _ in MODELS:
            folder = os.path.join(DATA_ROOT, f"outputs_{cfg_key}_{model_key}")
            per_repo = {}
            for fname, repo in REPO_FILES.items():
                path = os.path.join(folder, fname)
                if os.path.exists(path):
                    per_repo[repo] = pd.read_csv(path)
                else:
                    print(f"WARNING: missing {path}")
            data[(cfg_key, model_key)] = per_repo
    return data


DATA = load_all()


def wilson_ci(successes, total, z=1.96):
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1 + z ** 2 / total
    center = (p + z ** 2 / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z ** 2 / (4 * total)) / total) / denom
    return max(0.0, (center - margin) * 100), min(100.0, (center + margin) * 100)


def label_counts(df):
    vc = df['label'].value_counts()
    c = int(vc.get('CORRECT', 0))
    i = int(vc.get('INCORRECT', 0))
    n = int(vc.get('NO', 0))
    e = int(vc.get('ERROR', 0))
    return c, i, n, e


def esc(s):
    """Escape LaTeX-sensitive characters in plain text cells (none expected, kept defensive)."""
    return str(s).replace('_', '\\_').replace('%', '\\%')


# ---------------------------------------------------------------------------
# TABLE I: Overall effectiveness (Answer rate, Accuracy, Precision, repo-SD)
# ---------------------------------------------------------------------------
def build_table1():
    rows = []
    for cfg_key, cfg_name in CONFIGS:
        for model_key, model_name in MODELS:
            per_repo = DATA[(cfg_key, model_key)]
            tot_c = tot_i = tot_n = tot_e = 0
            repo_strict = []
            for repo in REPOS:
                df = per_repo.get(repo)
                if df is None:
                    continue
                c, i, n, e = label_counts(df)
                tot_c += c; tot_i += i; tot_n += n; tot_e += e
                total_repo = c + i + n + e
                repo_strict.append(100.0 * c / total_repo if total_repo else 0.0)
            total = tot_c + tot_i + tot_n + tot_e
            answer_rate = 100.0 * (tot_c + tot_i) / total if total else 0.0
            cond_acc = 100.0 * tot_c / (tot_c + tot_i) if (tot_c + tot_i) else 0.0
            strict_acc = 100.0 * tot_c / total if total else 0.0
            acc_lo, acc_hi = wilson_ci(tot_c, total)
            prec_lo, prec_hi = wilson_ci(tot_c, tot_c + tot_i) if (tot_c + tot_i) else (0.0, 0.0)
            std_strict = float(np.std(repo_strict, ddof=1)) if len(repo_strict) > 1 else 0.0
            rows.append({
                'Model': model_name, 'Config': cfg_name, 'N': total,
                'Correct': tot_c, 'Incorrect': tot_i, 'Refusals': tot_n, 'Errors': tot_e,
                'AnswerRate': round(answer_rate, 1),
                'Accuracy': round(strict_acc, 1), 'Accuracy_lo': round(acc_lo, 1), 'Accuracy_hi': round(acc_hi, 1),
                'Precision': round(cond_acc, 1), 'Precision_lo': round(prec_lo, 1), 'Precision_hi': round(prec_hi, 1),
                'SD_repo': round(std_strict, 1),
            })
    return pd.DataFrame(rows)


TABLE1 = build_table1()
TABLE1.to_csv(os.path.join(TABLE_DIR, "table1_overall_effectiveness.csv"), index=False)


def write_table1_tex():
    best = {}  # per model: config with best accuracy
    for model_name in [m[1] for m in MODELS]:
        sub = TABLE1[TABLE1['Model'] == model_name]
        best[model_name] = sub.loc[sub['Accuracy'].idxmax(), 'Config']

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{2pt}")
    lines.append(r"\caption{Answer rate, accuracy, and precision (\%), with Wilson 95\% intervals in brackets (KG: knowledge graph). Best accuracy and precision per model in bold.}")
    lines.append(r"\label{tab:accuracy-results}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Configuration} & \textbf{Answer rate} & \textbf{Accuracy} & \textbf{Precision} & \textbf{SD (repo)} \\")
    lines.append(r"\midrule")
    for mi, model_name in enumerate([m[1] for m in MODELS]):
        lines.append(r"\multicolumn{5}{l}{\textit{%s}} \\" % model_name)
        sub = TABLE1[TABLE1['Model'] == model_name]
        for _, row in sub.iterrows():
            is_best = row['Config'] == best[model_name]
            acc_str = f"{row['Accuracy']:.1f} [{row['Accuracy_lo']:.1f}, {row['Accuracy_hi']:.1f}]"
            prec_str = f"{row['Precision']:.1f} [{row['Precision_lo']:.1f}, {row['Precision_hi']:.1f}]"
            if is_best:
                acc_str = r"\textbf{%s}" % acc_str
                prec_str = r"\textbf{%s}" % prec_str
            lines.append(f"{row['Config']}   & {row['AnswerRate']:.1f}  & {acc_str} & {prec_str} & {row['SD_repo']:.1f} \\\\")
        if mi < len(MODELS) - 1:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    with open(os.path.join(TABLE_DIR, "table1_overall_effectiveness.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


write_table1_tex()


# ---------------------------------------------------------------------------
# TABLE II: Tier breakdown (Accuracy / Precision per tier)
# ---------------------------------------------------------------------------
def build_table2():
    rows = []
    for cfg_key, cfg_name in CONFIGS:
        for model_key, model_name in MODELS:
            per_repo = DATA[(cfg_key, model_key)]
            for tier_name, (lo, hi) in TIER_SLICES.items():
                tot_c = tot_i = tot_n = tot_e = 0
                for repo in REPOS:
                    df = per_repo.get(repo)
                    if df is None:
                        continue
                    sub = df.iloc[lo:hi]
                    c, i, n, e = label_counts(sub)
                    tot_c += c; tot_i += i; tot_n += n; tot_e += e
                total = tot_c + tot_i + tot_n + tot_e
                answer_rate = 100.0 * (tot_c + tot_i) / total if total else 0.0
                cond_acc = 100.0 * tot_c / (tot_c + tot_i) if (tot_c + tot_i) else 0.0
                strict_acc = 100.0 * tot_c / total if total else 0.0
                rows.append({
                    'Model': model_name, 'Config': cfg_name, 'Tier': tier_name, 'N': total,
                    'AnswerRate': round(answer_rate, 1),
                    'Precision': round(cond_acc, 1),
                    'Accuracy': round(strict_acc, 1),
                })
    return pd.DataFrame(rows)


TABLE2 = build_table2()
TABLE2.to_csv(os.path.join(TABLE_DIR, "table2_tier_breakdown.csv"), index=False)


def write_table2_tex():
    tiers = list(TIER_SLICES.keys())
    tier_short = {'Informational': 'Info.', 'Computational': 'Comp.', 'Machine Learning': 'ML'}
    ns = {'Informational': 100, 'Computational': 100, 'Machine Learning': 50}

    best = {}
    for model_name in [m[1] for m in MODELS]:
        for tier in tiers:
            sub = TABLE2[(TABLE2['Model'] == model_name) & (TABLE2['Tier'] == tier)]
            best[(model_name, tier)] = sub.loc[sub['Accuracy'].idxmax(), 'Config']

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(
        r"\caption{Accuracy / precision (\%%) per tier (Info.: %d questions, Comp.: %d, ML: %d), with the best accuracy per model in bold. Answer rate is accuracy divided by precision.}"
        % (ns['Informational'], ns['Computational'], ns['Machine Learning'])
    )
    lines.append(r"\label{tab:tier-results}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{3}{c}{\textbf{Accuracy / precision per tier}} \\")
    lines.append(r"\cmidrule(lr){2-4}")
    lines.append(r"\textbf{Configuration} & \textbf{%s} & \textbf{%s} & \textbf{%s} \\" %
                 (tier_short['Informational'], tier_short['Computational'], tier_short['Machine Learning']))
    lines.append(r"\midrule")
    for mi, model_name in enumerate([m[1] for m in MODELS]):
        lines.append(r"\multicolumn{4}{l}{\textit{%s}} \\" % model_name)
        for cfg_key, cfg_name in CONFIGS:
            cells = []
            for tier in tiers:
                row = TABLE2[(TABLE2['Model'] == model_name) & (TABLE2['Config'] == cfg_name) & (TABLE2['Tier'] == tier)].iloc[0]
                cell = f"{row['Accuracy']:.1f} / {row['Precision']:.1f}"
                if best[(model_name, tier)] == cfg_name:
                    cell = r"\textbf{%s}" % cell
                cells.append(cell)
            lines.append(f"{cfg_name}   & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
        if mi < len(MODELS) - 1:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    with open(os.path.join(TABLE_DIR, "table2_tier_breakdown.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


write_table2_tex()


# ---------------------------------------------------------------------------
# TABLE III: Cost / efficiency (now includes the KG configuration)
# ---------------------------------------------------------------------------
def build_table3():
    rows = []
    for cfg_key, cfg_name in CONFIGS:
        for model_key, model_name in MODELS:
            per_repo = DATA[(cfg_key, model_key)]
            tokens_all = []
            tools_all = []
            tot_c = 0
            for repo in REPOS:
                df = per_repo.get(repo)
                if df is None:
                    continue
                tokens_all.extend(df['token_count'].dropna().tolist())
                if 'tool_use' in df.columns:
                    tools_all.extend(df['tool_use'].dropna().tolist())
                c, _, _, _ = label_counts(df)
                tot_c += c
            tokens_all = np.array(tokens_all, dtype=float)
            total_tokens = tokens_all.sum()
            median_tok = float(np.median(tokens_all)) if len(tokens_all) else 0.0
            q25 = float(np.percentile(tokens_all, 25)) if len(tokens_all) else 0.0
            q75 = float(np.percentile(tokens_all, 75)) if len(tokens_all) else 0.0
            p95 = float(np.percentile(tokens_all, 95)) if len(tokens_all) else 0.0
            iqr = q75 - q25
            mean_tools = float(np.mean(tools_all)) if tools_all else 0.0
            sd_tools = float(np.std(tools_all, ddof=1)) if len(tools_all) > 1 else 0.0
            tokens_per_correct = total_tokens / tot_c if tot_c else float('nan')
            rows.append({
                'Model': model_name, 'Config': cfg_name,
                'MedianTok': round(median_tok, 0),
                'IQRTok': round(iqr, 0),
                'P95Tok': round(p95, 0),
                'MeanTools': round(mean_tools, 2),
                'SDTools': round(sd_tools, 2),
                'TotalTokens': int(total_tokens),
                'Correct': tot_c,
                'TokensPerCorrect': round(tokens_per_correct, 0) if tot_c else float('nan'),
            })
    return pd.DataFrame(rows)


TABLE3 = build_table3()
TABLE3.to_csv(os.path.join(TABLE_DIR, "table3_efficiency.csv"), index=False)


def fmt_int_thousands(x):
    return f"{int(round(x)):,}".replace(",", "{,}")


def write_table3_tex():
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\caption{Tokens per question (median, IQR, P95), tool calls per question (mean, SD), and tokens per correct answer (total over correct answers). Baseline tokens exclude the query-generation call. KG: knowledge graph.}")
    lines.append(r"\label{tab:cost-results}")
    lines.append(r"\begin{tabular}{lrrrcc@{\hspace{6pt}}c}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{3}{c}{\textbf{Tokens per question}} & \multicolumn{2}{c}{\textbf{Tool calls}} & \\")
    lines.append(r"\cmidrule(lr){2-4} \cmidrule(lr){5-6}")
    lines.append(r"\textbf{Configuration} & \textbf{Median} & \textbf{IQR} & \textbf{P95} & \textbf{Mean} & \textbf{SD} & \shortstack{\textbf{Tokens per}\\\textbf{correct answer}} \\")
    lines.append(r"\midrule")
    for mi, model_name in enumerate([m[1] for m in MODELS]):
        lines.append(r"\multicolumn{7}{l}{\textit{%s}} \\" % model_name)
        sub = TABLE3[TABLE3['Model'] == model_name]
        for _, row in sub.iterrows():
            mean_tools = "--" if row['MeanTools'] == 0 and row['Config'] == 'Baseline' else f"{row['MeanTools']:.2f}"
            sd_tools = "--" if row['SDTools'] == 0 and row['Config'] == 'Baseline' else f"{row['SDTools']:.2f}"
            lines.append(f"{row['Config']}   & {fmt_int_thousands(row['MedianTok'])}    & {fmt_int_thousands(row['IQRTok'])}     & {fmt_int_thousands(row['P95Tok'])}  & {mean_tools} & {sd_tools} & {fmt_int_thousands(row['TokensPerCorrect'])}  \\\\")
        if mi < len(MODELS) - 1:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    with open(os.path.join(TABLE_DIR, "table3_efficiency.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


write_table3_tex()


# ---------------------------------------------------------------------------
# TABLE IV (NEW): Per-repository Strict Accuracy, Mongo-only vs Full+KG
# Replaces the old per-repository bar/dumbbell figures with a compact table
# covering all three models at once.
# ---------------------------------------------------------------------------
def build_table4():
    rows = []
    for repo in REPOS:
        row = {'Repository': repo}
        for model_key, model_name in MODELS:
            for cfg_key, cfg_name in [('mongo_only', 'Mongo'), ('kg', 'KG')]:
                df = DATA[(cfg_key, model_key)].get(repo)
                if df is None:
                    row[f"{model_name}_{cfg_name}"] = float('nan')
                    continue
                c, i, n, e = label_counts(df)
                total = c + i + n + e
                row[f"{model_name}_{cfg_name}"] = round(100.0 * c / total, 0) if total else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


TABLE4 = build_table4()
TABLE4.to_csv(os.path.join(TABLE_DIR, "table4_per_repository.csv"), index=False)


def write_table4_tex():
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\caption{Per-repository Strict Accuracy (\% of 50 questions correct), Mongo-only (M) vs.\ Full+KG (K) configuration, for each model. Repository order: Kafka, Tomcat, Hibernate, Jenkins, Spring Boot.}")
    lines.append(r"\label{tab:per-repo}")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{2}{c}{\textbf{gpt-oss:20b}} & \multicolumn{2}{c}{\textbf{gpt-oss:120b}} \\")
    lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}")
    lines.append(r"\textbf{Repository} & \textbf{M} & \textbf{K} & \textbf{M} & \textbf{K} & \textbf{M} & \textbf{K} \\")
    lines.append(r"\midrule")
    for _, row in TABLE4.iterrows():
        lines.append(
            f"{row['Repository']}   "
            f"& {row['gpt-oss:20b_Mongo']:.0f}  & {row['gpt-oss:20b_KG']:.0f}  "
            f"& {row['gpt-oss:120b_Mongo']:.0f}  & {row['gpt-oss:120b_KG']:.0f}  \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    with open(os.path.join(TABLE_DIR, "table4_per_repository.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


write_table4_tex()


# ---------------------------------------------------------------------------
# FIGURE 1: Stacked bar (Correct / Incorrect / Refusal / Error)
# ---------------------------------------------------------------------------
def plot_figure1():
    cfg_names = [c[1] for c in CONFIGS]
    model_names = [m[1] for m in MODELS]

    fig, ax = plt.subplots(figsize=(8, 4.0))
    n_cfg = len(cfg_names)
    n_model = len(model_names)
    group_width = 0.8
    bar_width = group_width / n_model
    x = np.arange(n_cfg)

    for mi, (model_key, model_name) in enumerate(MODELS):
        correct_pct, refusal_pct, error_pct, incorrect_pct = [], [], [], []
        for cfg_key, cfg_name in CONFIGS:
            row = TABLE1[(TABLE1['Model'] == model_name) & (TABLE1['Config'] == cfg_name)].iloc[0]
            total = row['N']
            correct_pct.append(100.0 * row['Correct'] / total)
            incorrect_pct.append(100.0 * row['Incorrect'] / total)
            refusal_pct.append(100.0 * row['Refusals'] / total)
            error_pct.append(100.0 * row['Errors'] / total)

        offset = (mi - (n_model - 1) / 2) * bar_width
        xpos = x + offset

        lbl_correct = 'Correct' if mi == 0 else None
        lbl_incorrect = 'Incorrect' if mi == 0 else None
        lbl_refusal = 'Refusal' if mi == 0 else None
        lbl_error = 'Error' if mi == 0 else None

        ax.bar(xpos, correct_pct, bar_width * 0.9, color=STATUS_COLORS['Correct'],
               edgecolor='white', linewidth=0.8, label=lbl_correct)
        ax.bar(xpos, incorrect_pct, bar_width * 0.9, bottom=correct_pct,
               color=STATUS_COLORS['Incorrect'], edgecolor='white', linewidth=0.8, label=lbl_incorrect)
        bottom2 = np.array(correct_pct) + np.array(incorrect_pct)
        ax.bar(xpos, refusal_pct, bar_width * 0.9, bottom=bottom2,
               color=STATUS_COLORS['Refusal (NO)'], edgecolor='white', linewidth=0.8, label=lbl_refusal)
        bottom3 = bottom2 + np.array(refusal_pct)
        ax.bar(xpos, error_pct, bar_width * 0.9, bottom=bottom3,
               color=STATUS_COLORS['System Error'], edgecolor='white', linewidth=0.8, label=lbl_error)

        for xi in xpos:
            ax.text(xi, -1.0, model_name, ha='right', va='top', rotation=30)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_xticks(x)
    ax.set_xticklabels(cfg_names, rotation=0, va='top', y=-0.25)
    ax.set_ylabel('Share of Total Questions (%)', color='#111111')
    ax.set_ylim(0, 100)
    ax.set_yticklabels(range(0, 101, 20))
    ax.grid(axis='y', linestyle='--', alpha=0.4, color='#cccccc')
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_overall_effectiveness.pdf")
    plt.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


plot_figure1()


# ---------------------------------------------------------------------------
# FIGURE 2: Grouped bar - Strict Accuracy per tier (subplots), models on x-axis
# ---------------------------------------------------------------------------
def plot_figure2():
    tiers = list(TIER_SLICES.keys())
    cfg_names = [c[1] for c in CONFIGS]
    model_names = [m[1] for m in MODELS]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.0), sharey=True)

    for ax_i, tier in enumerate(tiers):
        ax = axes[ax_i]
        x = np.arange(len(model_names))
        n_cfg = len(cfg_names)
        width = 0.8 / n_cfg
        for ci, cfg_name in enumerate(cfg_names):
            vals = []
            for model_name in model_names:
                row = TABLE2[(TABLE2['Model'] == model_name) & (TABLE2['Config'] == cfg_name) & (TABLE2['Tier'] == tier)]
                vals.append(row['Accuracy'].values[0] if len(row) else 0)
            offset = (ci - (n_cfg - 1) / 2) * width
            ax.bar(x + offset, vals, width * 0.9, label=cfg_name, color=CONFIG_COLORS[cfg_name])
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        
        # Set tier text at the bottom of the subplot
        ax.text(0.5, -0.15, "(" + chr(ord('a') + ax_i) + ") " + tier, ha='center', va='top', fontsize=14, transform=ax.transAxes)

        ax.set_ylim(0, 100)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        if ax_i == 0:
            ax.set_ylabel('Strict Accuracy (%)')
        if ax_i == 2:
            ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0))

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_tier_scaling.pdf")
    plt.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


plot_figure2()


# ---------------------------------------------------------------------------
# FIGURE 3: Efficiency frontier (scatter) + token CDF
# ---------------------------------------------------------------------------
def plot_figure3():
    fig = plt.figure(figsize=(16, 4))
    gs = fig.add_gridspec(1, 2, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    for cfg_key, cfg_name in CONFIGS:
        for model_key, model_name in MODELS:
            t_row = TABLE3[(TABLE3['Model'] == model_name) & (TABLE3['Config'] == cfg_name)].iloc[0]
            a_row = TABLE1[(TABLE1['Model'] == model_name) & (TABLE1['Config'] == cfg_name)].iloc[0]
            x = t_row['MedianTok']
            y = a_row['Accuracy']
            ax1.scatter(x, y, s=140, color=MODEL_COLORS[model_name],
                        marker=CONFIG_MARKERS[cfg_name], edgecolor='black', linewidth=0.8, zorder=3)
            if cfg_name == 'Mongo-only' and model_name == 'gpt-oss:120b':
                ax1.annotate(f"{cfg_name}", (x, y), textcoords="offset points", xytext=(-50, 10))
            elif cfg_name == 'Full' and model_name == 'gpt-oss:120b':
                ax1.annotate(f"{cfg_name}", (x, y), textcoords="offset points", xytext=(-16, 6))
            elif cfg_name == 'Mongo-only' and model_name == 'gpt-oss:20b':
                ax1.annotate(f"{cfg_name}", (x, y), textcoords="offset points", xytext=(4, -16))
            elif cfg_name == 'Full' and model_name == 'gpt-oss:20b':
                ax1.annotate(f"{cfg_name}", (x, y), textcoords="offset points", xytext=(4, 4))
            elif cfg_name == 'KG' and model_name == 'gpt-oss:20b':
                ax1.annotate(f"{cfg_name}", (x, y), textcoords="offset points", xytext=(-16, 8))
            elif cfg_name == 'Mongo-only' and model_name == 'qwen2.5:7b':
                ax1.annotate(f"{cfg_name}", (x, y), textcoords="offset points", xytext=(4, 8))
            elif cfg_name == 'Full' and model_name == 'qwen2.5:7b':
                ax1.annotate(f"{cfg_name}", (x, y), textcoords="offset points", xytext=(4, -16))
            else:
                ax1.annotate(f"{cfg_name}", (x, y), textcoords="offset points", xytext=(6, 6))

    from matplotlib.lines import Line2D
    model_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10, label=m)
                      for m, c in MODEL_COLORS.items()]
    leg1 = ax1.legend(handles=model_handles, loc='lower right')
    ax1.add_artist(leg1)
    ax1.set_xlabel('Median Tokens per Question')
    ax1.set_ylabel('Strict Accuracy (%)')
    ax1.set_ylim(0, 100)
    ax1.text(0.5, -0.25, '(a) Cost\u2013Accuracy Trade-off Frontier', ha='center', va='top', transform=ax1.transAxes)
    ax1.grid(linestyle='--', alpha=0.5)

    for cfg_key, cfg_name in [('mongo_only', 'Mongo-only'), ('full', 'Full'), ('kg', 'KG')]:
        for model_key, model_name in MODELS:
            per_repo = DATA[(cfg_key, model_key)]
            toks = []
            for repo in REPOS:
                df = per_repo.get(repo)
                if df is not None:
                    toks.extend(df['token_count'].dropna().tolist())
            toks = np.sort(np.array(toks, dtype=float))
            if len(toks) == 0:
                continue
            yvals = np.arange(1, len(toks) + 1) / len(toks)
            linestyle = {'Mongo-only': ':', 'Full': '--', 'KG': '-'}[cfg_name]
            ax2.plot(toks, yvals, linestyle, color=MODEL_COLORS[model_name],
                      alpha=0.95 if cfg_name == 'KG' else 0.6,
                      linewidth=2 if cfg_name == 'KG' else 1.3,
                      label=f"{cfg_name} \u2013 {model_name}")
    ax2.set_xlabel('Tokens per Question')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_xlim(-10000, 400000)
    # set x axis to log scale
    ax2.text(0.5, -0.25, '(b) Token-Count CDF by Configuration', ha='center', va='top', transform=ax2.transAxes)
    ax2.grid(linestyle='--', alpha=0.5)
    ax2.legend(loc='lower right', ncol=1)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_efficiency_frontier.pdf")
    plt.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


plot_figure3()

print("\nAll paper tables & figures generated in:", BASE)
print(TABLE1.to_string(index=False))
print()
print(TABLE2.to_string(index=False))
print()
print(TABLE3.to_string(index=False))
print()
print(TABLE4.to_string(index=False))
