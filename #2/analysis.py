"""
=============================================================
BANA290 — AI Lab Assignment
RCT Analysis: AI PDF Extraction Tool on Loan Processing Clerks
Author : Alex (Yu-Wei Huang) | UCI NetID: yuweih13
Repo   : github.com/yuweih13-debug/AI-Lab
=============================================================

Prompt History (GitHub Copilot headers):
  Each section begins with the natural-language prompt used to
  direct Copilot. These appear as block comments labelled
  [Copilot Prompt].
"""

# ─────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────
import re
import requests
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from scipy import stats

warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════
# PHASE 1 — SCRAPE
# ═════════════════════════════════════════════════════════════

# [Copilot Prompt]
# "Scrape an HTML dashboard table from a given URL using
#  BeautifulSoup. Find the <table> element that contains
#  the columns CLERK, TREATMENT, TASKS_COMPLETED, and
#  ERROR_RATE. Parse every data row into a pandas DataFrame
#  and assign clean column names based on the known schema."

DATA_URL = "https://bana290-assignment2.netlify.app/"

COLUMN_NAMES = [
    "CLERK_COMBINED", "CLERK_ID", "QUEUE", "SITE", "SHIFT",
    "YEARS_EXPERIENCE", "BASELINE_TASKS_PER_HOUR",
    "BASELINE_ERROR_RATE", "TRAINING_SCORE", "TREATMENT",
    "SHIFT_START", "SHIFT_END", "TASKS_COMPLETED", "ERROR_RATE",
]

def scrape_dashboard(url: str) -> pd.DataFrame:
    """
    [Copilot Prompt]
    "Fetch the page with a browser-like User-Agent header,
     parse all <table> tags, select the one whose text
     contains 'TREATMENT' and 'TASKS_COMPLETED', then
     extract every non-header row into a list of lists and
     return a DataFrame."
    """
    hdrs = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"}
    resp = requests.get(url, headers=hdrs, timeout=20)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    target = None
    for tbl in soup.find_all("table"):
        txt = tbl.get_text()
        if "TREATMENT" in txt and "TASKS_COMPLETED" in txt:
            target = tbl
            break

    if target is None:
        raise ValueError("Target table not found on page.")

    rows = []
    for i, tr in enumerate(target.find_all("tr")):
        cells = [td.get_text(separator=" ", strip=True)
                 for td in tr.find_all(["th", "td"])]
        if i == 0:          # skip header row
            continue
        if len(cells) >= 2:
            rows.append(cells)

    max_w = max(len(r) for r in rows)
    rows  = [r + [""] * (max_w - len(r)) for r in rows]

    df = pd.DataFrame(rows).iloc[:, :len(COLUMN_NAMES)]
    df.columns = COLUMN_NAMES[:df.shape[1]]

    print(f"[SCRAPE] {len(df)} rows scraped from {url}")
    return df


df_raw = scrape_dashboard(DATA_URL)
print(df_raw[["CLERK_ID", "TREATMENT", "TASKS_COMPLETED",
              "ERROR_RATE", "SHIFT_START"]].head(5).to_string())


# ═════════════════════════════════════════════════════════════
# PHASE 2 — CLEAN
# ═════════════════════════════════════════════════════════════

df = df_raw.copy()

# ── 2a: Map TREATMENT labels to binary ───────────────────────

# [Copilot Prompt]
# "Create two predefined keyword lists — one for all treatment
#  label variants (AI Extract, Assist-On, Prefill Enabled,
#  Group B, Treatment) and one for control variants (Control,
#  Manual Entry, Typing Only, Group A, None). Write a function
#  that lowercases the input, checks membership in each list,
#  and returns 1 for treated, 0 for control, or None if the
#  label matches neither list so the row can be dropped."

TREATMENT_KEYWORDS = [
    "ai extract", "assist-on", "prefill enabled", "prefill",
    "group b", "treatment", "ai assist", "ai-enhanced",
]
CONTROL_KEYWORDS = [
    "control", "manual entry", "typing only", "group a",
    "no ai", "standard", "baseline", "manual", "none",
]

def map_treatment_label(label: str):
    """Return 1 (treated), 0 (control), or None (ambiguous)."""
    lo = str(label).lower().strip()
    if lo in ("none", "", "nan"):
        return 0
    for kw in TREATMENT_KEYWORDS:
        if kw in lo:
            return 1
    for kw in CONTROL_KEYWORDS:
        if kw in lo:
            return 0
    return None  # ambiguous — will be dropped


df["TREATMENT_BINARY"] = df["TREATMENT"].apply(map_treatment_label)
n_ambiguous = df["TREATMENT_BINARY"].isna().sum()
df.dropna(subset=["TREATMENT_BINARY"], inplace=True)
df["TREATMENT_BINARY"] = df["TREATMENT_BINARY"].astype(int)

print(f"\n[CLEAN] Dropped {n_ambiguous} ambiguous TREATMENT rows.")
print(f"[CLEAN] Treated: {df.TREATMENT_BINARY.sum()} | "
      f"Control: {(df.TREATMENT_BINARY == 0).sum()}")


# ── 2b: Extract numeric values from text-embedded columns ────

# [Copilot Prompt]
# "Use re.sub(r'[^\\d.]', '', text) to strip all non-numeric
#  characters from TASKS_COMPLETED, ERROR_RATE, YEARS_EXPERIENCE,
#  and BASELINE_ERROR_RATE. For TRAINING_SCORE, values like
#  '89/100' and 'score 87' should yield just the score integer —
#  take the numerator of any fraction. Return NaN for 'TBD',
#  'pending log', '--', and other non-numeric placeholders."

def extract_numeric(text: str) -> float:
    """Strip units/labels and return a float, or NaN."""
    s = str(text).strip()
    if s.lower() in ("tbd", "n/a", "", "nan", "pending log", "--"):
        return np.nan
    cleaned = re.sub(r"[^\d.]", "", s)
    parts = cleaned.split(".")
    if len(parts) > 2:
        cleaned = parts[0] + "." + parts[1]
    try:
        return float(cleaned) if cleaned else np.nan
    except ValueError:
        return np.nan


def extract_score(text: str) -> float:
    """
    [Copilot Prompt]
    "Parse training score strings like '89/100', 'score 87',
     '85 pts', '95/100' — always return only the score value,
     not the denominator."
    """
    s = str(text).strip()
    if s.lower() in ("tbd", "n/a", "", "nan", "pending log", "--"):
        return np.nan
    s_clean = re.sub(r"(?i)(score|pts|points|\s)", "", s)
    if "/" in s_clean:
        return float(s_clean.split("/")[0])
    c = re.sub(r"[^\d.]", "", s_clean)
    try:
        return float(c) if c else np.nan
    except ValueError:
        return np.nan


for col in ["TASKS_COMPLETED", "ERROR_RATE",
            "YEARS_EXPERIENCE", "BASELINE_ERROR_RATE"]:
    df[col] = df[col].apply(extract_numeric)

df["TRAINING_SCORE"] = df["TRAINING_SCORE"].apply(extract_score)

print(f"[CLEAN] Numeric extraction complete.")
print(f"        Nulls — TASKS: {df.TASKS_COMPLETED.isna().sum()}, "
      f"ERROR: {df.ERROR_RATE.isna().sum()}, "
      f"TRAINING_SCORE range: {df.TRAINING_SCORE.min():.0f}–"
      f"{df.TRAINING_SCORE.max():.0f}")


# ── 2c: Parse mixed-format timestamps → shift duration ───────

# [Copilot Prompt]
# "SHIFT_START and SHIFT_END contain wildly inconsistent formats
#  such as 'Feb 18, 2026 07:56', '2026-02-18 15:50',
#  '21-Feb-2026 08:19 AM', and '02/21/2026 04:26 PM'.
#  Use pd.to_datetime() on both columns, then subtract them to
#  get the exact shift duration in decimal hours. Replace
#  'pending log' and '--' with NaT before parsing."

def parse_timestamps(series: pd.Series) -> pd.Series:
    """
    [Copilot Prompt]
    "Replace known garbage strings with None, then call
     pd.to_datetime with format='mixed' and errors='coerce'
     to handle all valid timestamp variants in one pass."
    """
    cleaned = series.str.strip().replace(
        {"pending log": None, "--": None, "TBD": None, "": None}
    )
    return pd.to_datetime(cleaned, format="mixed", errors="coerce")


df["SHIFT_START_DT"] = parse_timestamps(df["SHIFT_START"])
df["SHIFT_END_DT"]   = parse_timestamps(df["SHIFT_END"])

df["SHIFT_DURATION_HRS"] = (
    (df["SHIFT_END_DT"] - df["SHIFT_START_DT"])
    .dt.total_seconds() / 3600
)

# Invalidate implausible durations (< 4 hrs or > 14 hrs)
df.loc[
    (df["SHIFT_DURATION_HRS"] < 4) | (df["SHIFT_DURATION_HRS"] > 14),
    "SHIFT_DURATION_HRS"
] = np.nan

n_bad_ts = df["SHIFT_DURATION_HRS"].isna().sum()
print(f"[CLEAN] Timestamp parsing done. "
      f"{n_bad_ts} rows with missing/implausible shift duration.")


# ── 2d: Final cleaning and null removal ──────────────────────

# [Copilot Prompt]
# "Drop any row missing a value in the core analysis columns
#  TASKS_COMPLETED, ERROR_RATE, YEARS_EXPERIENCE, TRAINING_SCORE,
#  or TREATMENT_BINARY. Report how many rows were dropped,
#  then reset the index."

CORE_COLS = [
    "TASKS_COMPLETED", "ERROR_RATE",
    "YEARS_EXPERIENCE", "TRAINING_SCORE", "TREATMENT_BINARY",
]

before = len(df)
df.dropna(subset=CORE_COLS, inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"[CLEAN] Dropped {before - len(df)} rows with missing core values.")
print(f"[CLEAN] Final analytic dataset: {len(df)} rows.")
print("\n── Descriptive Statistics ────────────────────────────────")
print(df[CORE_COLS + ["SHIFT_DURATION_HRS"]].describe().round(2).to_string())


# ═════════════════════════════════════════════════════════════
# PHASE 3 — ANALYZE
# ═════════════════════════════════════════════════════════════

treat = df[df["TREATMENT_BINARY"] == 1]
ctrl  = df[df["TREATMENT_BINARY"] == 0]
print(f"\n[ANALYZE] Treatment n={len(treat)}, Control n={len(ctrl)}")


# ── 3a: Balance Test ─────────────────────────────────────────

# [Copilot Prompt]
# "Run Welch's two-sample t-tests on the pre-treatment covariates
#  YEARS_EXPERIENCE and TRAINING_SCORE to verify that random
#  assignment produced comparable groups. Return a formatted
#  DataFrame showing group means, standard deviations,
#  t-statistic, p-value, and a PASS/FAIL balance flag."

def balance_test(df: pd.DataFrame, covariates: list) -> pd.DataFrame:
    rows = []
    for col in covariates:
        t_v = df.loc[df.TREATMENT_BINARY == 1, col].dropna()
        c_v = df.loc[df.TREATMENT_BINARY == 0, col].dropna()
        t_stat, p_val = stats.ttest_ind(t_v, c_v, equal_var=False)
        rows.append({
            "Covariate":    col,
            "Treat Mean":   round(t_v.mean(), 2),
            "Treat SD":     round(t_v.std(),  2),
            "Control Mean": round(c_v.mean(), 2),
            "Control SD":   round(c_v.std(),  2),
            "t-stat":       round(t_stat, 3),
            "p-value":      round(p_val,  4),
            "Balanced":     "PASS" if p_val > 0.1 else "FAIL",
        })
    return pd.DataFrame(rows)


balance_df = balance_test(df, ["YEARS_EXPERIENCE", "TRAINING_SCORE"])
print("\n── Balance Test ──────────────────────────────────────────")
print(balance_df.to_string(index=False))


# ── 3b: Ignorability assumption ──────────────────────────────

# [Copilot Prompt]
# "Formally verify the ignorability assumption by checking
#  whether any pre-treatment covariate differs significantly
#  between treatment and control groups. Print a PASS if
#  p > 0.10, FAIL otherwise, and explain what each result
#  implies for the validity of the RCT."

print("\n── Ignorability Assumption Test ──────────────────────────")
for _, row in balance_df.iterrows():
    flag = "PASS — no significant pre-treatment difference" \
           if row["p-value"] > 0.1 \
           else "FAIL — potential confound"
    print(f"  {row['Covariate']:<22}  p = {row['p-value']:.4f}  [{flag}]")


# ── 3c: ATE Estimation ───────────────────────────────────────

# [Copilot Prompt]
# "Compute the Average Treatment Effect using the
#  difference-in-means estimator for both TASKS_COMPLETED
#  (quantity outcome) and ERROR_RATE (quality outcome).
#  Use Welch's t-test to assess significance. Return the ATE,
#  standard error, t-statistic, p-value, and 95% confidence
#  interval for each outcome."

def estimate_ate(df: pd.DataFrame, outcome: str) -> dict:
    t_v = df.loc[df.TREATMENT_BINARY == 1, outcome].dropna()
    c_v = df.loc[df.TREATMENT_BINARY == 0, outcome].dropna()
    ate    = t_v.mean() - c_v.mean()
    se     = np.sqrt(t_v.var(ddof=1)/len(t_v) + c_v.var(ddof=1)/len(c_v))
    t_stat, p_val = stats.ttest_ind(t_v, c_v, equal_var=False)
    return {
        "Outcome":    outcome,
        "n_T":        len(t_v),
        "n_C":        len(c_v),
        "Treat Mean": round(t_v.mean(), 2),
        "Ctrl Mean":  round(c_v.mean(), 2),
        "ATE":        round(ate, 3),
        "SE":         round(se,  3),
        "t-stat":     round(t_stat, 3),
        "p-value":    round(p_val,  4),
        "CI_lo":      round(ate - 1.96 * se, 3),
        "CI_hi":      round(ate + 1.96 * se, 3),
    }


ate_tasks  = estimate_ate(df, "TASKS_COMPLETED")
ate_errors = estimate_ate(df, "ERROR_RATE")

print("\n── ATE Estimation Results ────────────────────────────────")
for r in [ate_tasks, ate_errors]:
    print(f"\n  Outcome       : {r['Outcome']}")
    print(f"  n (T / C)     : {r['n_T']} / {r['n_C']}")
    print(f"  Treat Mean    : {r['Treat Mean']}")
    print(f"  Control Mean  : {r['Ctrl Mean']}")
    print(f"  ATE           : {r['ATE']}")
    print(f"  SE            : {r['SE']}")
    print(f"  t-statistic   : {r['t-stat']}")
    print(f"  p-value       : {r['p-value']}")
    print(f"  95% CI        : [{r['CI_lo']}, {r['CI_hi']}]")


# ═════════════════════════════════════════════════════════════
# PHASE 4 — FIGURES & TABLES (exported for LaTeX)
# ═════════════════════════════════════════════════════════════

# [Copilot Prompt]
# "Create a publication-quality 1x2 figure with side-by-side
#  boxplots comparing treatment vs. control for TASKS_COMPLETED
#  and ERROR_RATE. Use a whitegrid seaborn theme, distinct colors
#  for each group, annotate each plot with the ATE and p-value
#  in a shaded text box, and save as both PDF and PNG."

sns.set_theme(style="whitegrid", font_scale=1.15)
PALETTE = {"Control": "#4C72B0", "Treatment": "#DD8452"}
df["Group"] = df["TREATMENT_BINARY"].map({1: "Treatment", 0: "Control"})

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

plot_specs = [
    ("TASKS_COMPLETED", "Productivity: Tasks Completed per Shift",
     "Tasks Completed", ate_tasks),
    ("ERROR_RATE", "Quality: Error Rate per Shift",
     "Error Rate (%)", ate_errors),
]

for ax, (col, title, ylabel, res) in zip(axes, plot_specs):
    sns.boxplot(
        data=df, x="Group", y=col, palette=PALETTE,
        width=0.45, linewidth=1.5,
        order=["Control", "Treatment"], ax=ax,
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
    )
    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    sign  = "+" if res["ATE"] > 0 else ""
    p_str = "p < 0.001" if res["p-value"] < 0.001 \
            else f"p = {res['p-value']:.3f}"
    ax.annotate(
        f"ATE = {sign}{res['ATE']}\n{p_str}",
        xy=(0.5, 0.96), xycoords="axes fraction",
        ha="center", va="top", fontsize=10, color="#8B0000",
        bbox=dict(boxstyle="round,pad=0.35",
                  fc="lightyellow", ec="#aaa", alpha=0.9),
    )

fig.suptitle(
    "RCT Results — AI PDF Extraction Tool vs. Manual Processing\n"
    "Loan Operations Clerks, March 2026 Audit Week",
    fontsize=12, fontweight="bold", y=1.02,
)
plt.tight_layout()
plt.savefig("boxplots.pdf", bbox_inches="tight", dpi=300)
plt.savefig("boxplots.png", bbox_inches="tight", dpi=150)
plt.close()
print("\n[OUTPUT] Saved boxplots.pdf and boxplots.png")


# [Copilot Prompt]
# "Export the balance table and ATE results table as
#  booktabs-formatted LaTeX using pandas to_latex(), with
#  proper captions and labels, saved to tables.tex for
#  inclusion in the interpretation write-up."

balance_export = balance_df[
    ["Covariate", "Treat Mean", "Control Mean", "t-stat", "p-value"]
]

at2 = dict(ate_tasks);  at2["95% CI"] = f"[{at2.pop('CI_lo')}, {at2.pop('CI_hi')}]"
ae2 = dict(ate_errors); ae2["95% CI"] = f"[{ae2.pop('CI_lo')}, {ae2.pop('CI_hi')}]"
ate_export = pd.DataFrame([at2, ae2])[
    ["Outcome", "Treat Mean", "Ctrl Mean", "ATE",
     "SE", "t-stat", "p-value", "95% CI"]
]

with open("tables.tex", "w") as f:
    f.write("% ── Balance Table ──────────────────────────────────\n")
    f.write(balance_export.to_latex(
        index=False, escape=True, column_format="lrrrr",
        caption="Pre-Treatment Balance Test (Welch's $t$-test)",
        label="tab:balance", position="H",
    ))
    f.write("\n\n% ── ATE Results Table ──────────────────────────────\n")
    f.write(ate_export.to_latex(
        index=False, escape=True, column_format="lrrrrrrl",
        caption="Average Treatment Effect Estimates",
        label="tab:ate", position="H",
    ))

print("[OUTPUT] Saved tables.tex")
print("\n[DONE] All four phases complete. "
      "Outputs: boxplots.pdf, boxplots.png, tables.tex")
# Scrape phase complete
# Clean phase complete
