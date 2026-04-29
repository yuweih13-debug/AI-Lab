"""
=============================================================
BANA290 — AI Lab Assignment 4
IV + RDD Analysis: AI Infrastructure and Innovation
Smart Campus Student App Incubator Archive
Author : Alex (Yu-Wei Huang) | UCI NetID: yuweih13
Repo   : github.com/yuweih13-debug/AI-Lab

Prompt History (GitHub Copilot headers):
  Each section begins with the natural-language prompt used
  to direct Copilot, labelled [Copilot Prompt].
=============================================================
"""

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
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.gmm import IV2SLS

warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════
# PHASE 1 — SCRAPE
# ═════════════════════════════════════════════════════════════

# [Copilot Prompt]
# "Scrape the index page of the incubator archive portal using
#  BeautifulSoup. Extract all href links to individual brief
#  pages. Then loop through each URL and scrape its HTML table
#  into a separate DataFrame. Store them in a dictionary keyed
#  by page name and merge them using pd.merge() on TEAM_REF."

INDEX_URL = "https://bana290-assignment4.netlify.app/"
HEADERS   = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

def scrape_index(url: str) -> list:
    """
    [Copilot Prompt]
    "Fetch the index page, find all <a> tags whose href
     contains '/briefs/', deduplicate and return as a list
     of absolute URLs."
    """
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/briefs/" in href:
            full = href if href.startswith("http") else \
                   "https://bana290-assignment4.netlify.app" + href
            if full not in links:
                links.append(full)
    print(f"[SCRAPE] Found {len(links)} brief pages on index.")
    return links


def scrape_table(url: str) -> pd.DataFrame:
    """
    [Copilot Prompt]
    "Fetch a single brief page, locate the <table> element,
     extract headers from the first row and data from remaining
     rows, return as a pandas DataFrame with a source column."
    """
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table")
    if table is None:
        raise ValueError(f"No table at {url}")
    rows, hdrs = [], []
    for i, tr in enumerate(table.find_all("tr")):
        cells = [td.get_text(separator=" ", strip=True)
                 for td in tr.find_all(["th", "td"])]
        if i == 0:
            hdrs = cells
        elif cells:
            rows.append(cells)
    max_w = max(len(r) for r in rows)
    rows  = [r + [""] * (max_w - len(r)) for r in rows]
    df = pd.DataFrame(rows, columns=hdrs[:max_w])
    print(f"[SCRAPE]   {url.split('/')[-1]}: {len(df)} rows")
    return df


# Run scrape
brief_urls = scrape_index(INDEX_URL)
tables = {}
for url in brief_urls:
    key = url.split("/")[-1]
    tables[key] = scrape_table(url)

df_infra   = tables["fiber-access-bulletin"]
df_metrics = tables["builder-metrics-ledger"]
df_grants  = tables["anteater-fund-panel"]

print(f"[SCRAPE] Tables: infra={len(df_infra)}, "
      f"metrics={len(df_metrics)}, grants={len(df_grants)}")


# ═════════════════════════════════════════════════════════════
# PHASE 2 — CLEAN
# ═════════════════════════════════════════════════════════════

# ── 2a: Clean TEAM_REF (extract SC00X code only) ────────────

# [Copilot Prompt]
# "The TEAM_REF column contains a nested label like 'SC001
#  Anteater Cart'. Extract only the SC-prefixed identifier
#  using regex so all three tables can be merged on it."

def extract_team_ref(text: str) -> str:
    m = re.search(r"SC\d+", str(text))
    return m.group(0) if m else str(text).strip()

for df in [df_infra, df_metrics, df_grants]:
    df["TEAM_REF"] = df["TEAM_REF"].apply(extract_team_ref)


# ── 2b: Parse DISTANCE_TO_NODE → float (km) ─────────────────

# [Copilot Prompt]
# "DISTANCE_TO_NODE contains messy formats: '201 m',
#  '0.33 km from backbone', '~0.59km', 'Distance: 827 meters',
#  '0.81 km | fiber hop', '1,100 meters (sync route)'.
#  Write a function that extracts the numeric value and
#  converts meters to km so all values are in the same unit
#  as a float."

def parse_distance_km(text: str) -> float:
    s = str(text).lower().replace(",", "")
    num = re.search(r"[\d.]+", s)
    if not num:
        return np.nan
    val = float(num.group(0))
    # If km anywhere in string, it's already km
    if "km" in s:
        return round(val, 4)
    # Otherwise it's in meters
    return round(val / 1000, 4)

df_infra["DISTANCE_KM"] = df_infra["DISTANCE_TO_NODE"].apply(parse_distance_km)
print(f"\n[CLEAN] Distance range: {df_infra.DISTANCE_KM.min():.3f} – "
      f"{df_infra.DISTANCE_KM.max():.3f} km")


# ── 2c: Parse AI_INTENSITY → float ──────────────────────────

# [Copilot Prompt]
# "AI_INTENSITY uses formats like '52.4 gpu-hrs/wk',
#  '~57.1 model hrs', '62.9 builder hours'. Strip all
#  non-numeric characters except the decimal point and
#  return a float."

def parse_numeric(text: str) -> float:
    s = str(text).replace(",", "")
    m = re.search(r"[\d.]+", s)
    return float(m.group(0)) if m else np.nan

df_metrics["AI_INTENSITY"] = df_metrics["AI_INTENSITY"].apply(parse_numeric)


# ── 2d: Parse INNOVATION_SCORE → float ──────────────────────

# [Copilot Prompt]
# "INNOVATION_SCORE uses formats like '61.0 / 100',
#  '57.9 score', '73.7 / 100'. Extract only the first
#  numeric value before any slash or label text."

def parse_score(text: str) -> float:
    m = re.search(r"[\d.]+", str(text))
    return float(m.group(0)) if m else np.nan

df_metrics["INNOVATION_SCORE"] = df_metrics["INNOVATION_SCORE"].apply(parse_score)


# ── 2e: Parse ELIGIBILITY_SCORE → float ─────────────────────

# [Copilot Prompt]
# "ELIGIBILITY_SCORE uses formats like '82.9 / 100',
#  'Pitch rating = 81.0', '92.1 score', 'panel avg 89.5',
#  '80.9 points', 'Score: 80.4 pts'. Extract the numeric
#  value in all cases."

df_grants["ELIGIBILITY_SCORE"] = df_grants["ELIGIBILITY_SCORE"].apply(parse_score)


# ── 2f: Merge all three tables on TEAM_REF ──────────────────

# [Copilot Prompt]
# "Merge the infrastructure, metrics, and grants DataFrames
#  on TEAM_REF using pd.merge() with inner joins. Keep only
#  the columns needed for analysis."

df = (df_infra[["TEAM_REF","HOME_BASE","NETWORK_ZONE","DISTANCE_KM"]]
      .merge(df_metrics[["TEAM_REF","TRACK","AI_INTENSITY","INNOVATION_SCORE"]],
             on="TEAM_REF", how="inner")
      .merge(df_grants[["TEAM_REF","ELIGIBILITY_SCORE"]],
             on="TEAM_REF", how="inner"))

print(f"[CLEAN] Merged dataset: {len(df)} rows × {len(df.columns)} columns")


# ── 2g: Handle outliers with IQR capping ────────────────────

# [Copilot Prompt]
# "Use the IQR method to cap outliers in AI_INTENSITY and
#  INNOVATION_SCORE. Calculate Q1 and Q3, compute IQR,
#  then clip values to [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
#  This prevents extreme teams from skewing the regression."

def iqr_clip(series: pd.Series) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return series.clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

df["AI_INTENSITY"]     = iqr_clip(df["AI_INTENSITY"])
df["INNOVATION_SCORE"] = iqr_clip(df["INNOVATION_SCORE"])

df.dropna(subset=["DISTANCE_KM","AI_INTENSITY",
                  "INNOVATION_SCORE","ELIGIBILITY_SCORE"],
          inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"[CLEAN] Final dataset: {len(df)} rows")
print(df[["TEAM_REF","DISTANCE_KM","AI_INTENSITY",
          "INNOVATION_SCORE","ELIGIBILITY_SCORE"]].describe().round(2).to_string())


# ═════════════════════════════════════════════════════════════
# PHASE 3 — ANALYZE
# ═════════════════════════════════════════════════════════════

sns.set_theme(style="whitegrid", font_scale=1.1)
CUTOFF = 85.0

# ── 3a: Naive OLS (benchmark) ────────────────────────────────

# [Copilot Prompt]
# "Run a naive OLS regression of INNOVATION_SCORE on
#  AI_INTENSITY as a baseline benchmark. This will likely
#  be upward biased due to omitted variable bias."

ols_model  = smf.ols("INNOVATION_SCORE ~ AI_INTENSITY", data=df).fit()
ols_coef   = ols_model.params["AI_INTENSITY"]
ols_p      = ols_model.pvalues["AI_INTENSITY"]
print(f"\n── Naive OLS ─────────────────────────────────────────────")
print(f"  AI_INTENSITY coef: {ols_coef:.4f}  p={ols_p:.4f}")


# ── 3b: IV First Stage ───────────────────────────────────────

# [Copilot Prompt]
# "Run the first stage of 2SLS: regress AI_INTENSITY on
#  DISTANCE_KM (the instrument). Report the F-statistic
#  to test instrument relevance. A strong instrument has
#  F > 10."

first_stage = smf.ols("AI_INTENSITY ~ DISTANCE_KM", data=df).fit()
fs_coef     = first_stage.params["DISTANCE_KM"]
fs_fstat    = first_stage.fvalue
fs_p        = first_stage.pvalues["DISTANCE_KM"]
df["AI_INTENSITY_HAT"] = first_stage.fittedvalues

print(f"\n── IV First Stage ────────────────────────────────────────")
print(f"  DISTANCE_KM coef  : {fs_coef:.4f}")
print(f"  F-statistic       : {fs_fstat:.2f}")
print(f"  p-value           : {fs_p:.4f}")
print(f"  Instrument strength: {'STRONG (F>10)' if fs_fstat>10 else 'WEAK (F<10)'}")


# ── 3c: IV Second Stage (2SLS) ───────────────────────────────

# [Copilot Prompt]
# "Run the second stage of 2SLS: regress INNOVATION_SCORE
#  on the fitted values AI_INTENSITY_HAT from the first
#  stage. This gives the causal IV estimate."

second_stage = smf.ols("INNOVATION_SCORE ~ AI_INTENSITY_HAT", data=df).fit()
iv_coef      = second_stage.params["AI_INTENSITY_HAT"]
iv_p         = second_stage.pvalues["AI_INTENSITY_HAT"]
iv_ci        = second_stage.conf_int().loc["AI_INTENSITY_HAT"]

print(f"\n── IV Second Stage (2SLS) ────────────────────────────────")
print(f"  2SLS coefficient  : {iv_coef:.4f}")
print(f"  p-value           : {iv_p:.4f}")
print(f"  95% CI            : [{iv_ci.iloc[0]:.4f}, {iv_ci.iloc[1]:.4f}]")
print(f"\n  OLS vs IV comparison:")
print(f"    Naive OLS coef  : {ols_coef:.4f}")
print(f"    IV 2SLS coef    : {iv_coef:.4f}")
print(f"    Direction: {'IV > OLS (attenuation bias corrected)' if iv_coef > ols_coef else 'OLS > IV (OLS upward biased)'}")


# ── 3d: RDD — Visual Test ────────────────────────────────────

# [Copilot Prompt]
# "Plot INNOVATION_SCORE against ELIGIBILITY_SCORE with a
#  vertical dashed line at the cutoff (85 points). Add a
#  linear fit line on each side of the cutoff separately
#  to visualize the discontinuity jump. Save as rdd_plot.pdf."

fig, ax = plt.subplots(figsize=(10, 5.5))

below = df[df["ELIGIBILITY_SCORE"] < CUTOFF]
above = df[df["ELIGIBILITY_SCORE"] >= CUTOFF]

ax.scatter(below["ELIGIBILITY_SCORE"], below["INNOVATION_SCORE"],
           color="#4C72B0", alpha=0.7, label="Below cutoff (no grant)", s=55)
ax.scatter(above["ELIGIBILITY_SCORE"], above["INNOVATION_SCORE"],
           color="#DD8452", alpha=0.7, label="Above cutoff (grant awarded)", s=55)

# Fit lines on each side
for sub, color in [(below, "#4C72B0"), (above, "#DD8452")]:
    if len(sub) > 1:
        z = np.polyfit(sub["ELIGIBILITY_SCORE"], sub["INNOVATION_SCORE"], 1)
        p = np.poly1d(z)
        xs = np.linspace(sub["ELIGIBILITY_SCORE"].min(),
                         sub["ELIGIBILITY_SCORE"].max(), 100)
        ax.plot(xs, p(xs), color=color, linewidth=2)

ax.axvline(CUTOFF, color="crimson", linestyle="--",
           linewidth=2, label=f"RDD Cutoff = {CUTOFF}")
ax.set_title("RDD Visual Test: Innovation Score vs Eligibility Score",
             fontweight="bold")
ax.set_xlabel("Eligibility Score (Running Variable)")
ax.set_ylabel("Innovation Score")
ax.legend()
plt.tight_layout()
plt.savefig("rdd_plot.pdf", bbox_inches="tight", dpi=300)
plt.savefig("rdd_plot.png", bbox_inches="tight", dpi=150)
plt.close()
print("\n[OUTPUT] Saved rdd_plot.pdf")


# ── 3e: RDD Estimation ───────────────────────────────────────

# [Copilot Prompt]
# "Estimate the RDD treatment effect by running a local
#  linear regression with a bandwidth of 10 points around
#  the cutoff (75 to 95). Create an ABOVE_CUTOFF dummy and
#  interact it with the centered running variable."

bw = 10
rdd_df = df[(df["ELIGIBILITY_SCORE"] >= CUTOFF - bw) &
            (df["ELIGIBILITY_SCORE"] <= CUTOFF + bw)].copy()
rdd_df["ABOVE"]   = (rdd_df["ELIGIBILITY_SCORE"] >= CUTOFF).astype(int)
rdd_df["SCORE_C"] = rdd_df["ELIGIBILITY_SCORE"] - CUTOFF

rdd_model = smf.ols(
    "INNOVATION_SCORE ~ ABOVE + SCORE_C + ABOVE:SCORE_C",
    data=rdd_df
).fit()

rdd_coef = rdd_model.params["ABOVE"]
rdd_p    = rdd_model.pvalues["ABOVE"]
rdd_ci   = rdd_model.conf_int().loc["ABOVE"]

print(f"\n── RDD Local Linear Regression (BW={bw}) ─────────────────")
print(f"  Jump at cutoff (ABOVE coef): {rdd_coef:.4f}")
print(f"  p-value                    : {rdd_p:.4f}")
print(f"  95% CI                     : [{rdd_ci.iloc[0]:.4f}, {rdd_ci.iloc[1]:.4f}]")
print(f"  N in bandwidth             : {len(rdd_df)}")


# ── 3f: RDD Density / Manipulation Check ─────────────────────

# [Copilot Prompt]
# "Plot a histogram of ELIGIBILITY_SCORE to check for
#  manipulation (bunching) just above the 85-point cutoff.
#  If teams could manipulate their score, we would see an
#  unusual spike just above the threshold."

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.hist(df["ELIGIBILITY_SCORE"], bins=20, color="#4C72B0",
        edgecolor="white", alpha=0.85)
ax.axvline(CUTOFF, color="crimson", linestyle="--",
           linewidth=2, label=f"Cutoff = {CUTOFF}")
ax.set_title("Density Check: Distribution of Eligibility Scores",
             fontweight="bold")
ax.set_xlabel("Eligibility Score")
ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
plt.savefig("density_plot.pdf", bbox_inches="tight", dpi=300)
plt.savefig("density_plot.png", bbox_inches="tight", dpi=150)
plt.close()
print("[OUTPUT] Saved density_plot.pdf")


# ── 3g: First Stage scatter plot ─────────────────────────────

# [Copilot Prompt]
# "Plot AI_INTENSITY against DISTANCE_KM with a fitted
#  regression line to visualize the first-stage relationship.
#  Annotate with the F-statistic. Save as first_stage.pdf."

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(df["DISTANCE_KM"], df["AI_INTENSITY"],
           color="#4C72B0", alpha=0.7, s=55)
z = np.polyfit(df["DISTANCE_KM"], df["AI_INTENSITY"], 1)
xs = np.linspace(df["DISTANCE_KM"].min(), df["DISTANCE_KM"].max(), 100)
ax.plot(xs, np.poly1d(z)(xs), color="#DD8452", linewidth=2.2)
ax.set_title("IV First Stage: AI Intensity vs Distance to Fiber Node",
             fontweight="bold")
ax.set_xlabel("Distance to Fiber Node (km)")
ax.set_ylabel("AI Intensity (compute hrs/wk)")
ax.annotate(f"F-stat = {fs_fstat:.1f}  p = {fs_p:.4f}",
            xy=(0.05, 0.92), xycoords="axes fraction",
            fontsize=10, color="#8B0000",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="#aaa"))
plt.tight_layout()
plt.savefig("first_stage.pdf", bbox_inches="tight", dpi=300)
plt.savefig("first_stage.png", bbox_inches="tight", dpi=150)
plt.close()
print("[OUTPUT] Saved first_stage.pdf")


# ── 3h: Export LaTeX tables ──────────────────────────────────

# [Copilot Prompt]
# "Export the first stage, naive OLS, 2SLS, and RDD results
#  as a single LaTeX table file using pandas to_latex()
#  with booktabs formatting."

summary_df = pd.DataFrame([
    {"Model": "Naive OLS",    "Outcome": "Innovation Score",
     "Key Regressor": "AI Intensity",
     "Coefficient": round(ols_coef, 4), "p-value": round(ols_p, 4),
     "Note": "Upward biased"},
    {"Model": "IV First Stage", "Outcome": "AI Intensity",
     "Key Regressor": "Distance (km)",
     "Coefficient": round(fs_coef, 4), "p-value": round(fs_p, 4),
     "Note": f"F={fs_fstat:.1f}"},
    {"Model": "IV 2SLS",     "Outcome": "Innovation Score",
     "Key Regressor": "AI Intensity (IV)",
     "Coefficient": round(iv_coef, 4), "p-value": round(iv_p, 4),
     "Note": f"CI [{iv_ci.iloc[0]:.3f}, {iv_ci.iloc[1]:.3f}]"},
    {"Model": "RDD",         "Outcome": "Innovation Score",
     "Key Regressor": "Above Cutoff (85)",
     "Coefficient": round(rdd_coef, 4), "p-value": round(rdd_p, 4),
     "Note": f"BW={bw}, N={len(rdd_df)}"},
])

with open("tables.tex", "w") as f:
    f.write("% ── Main Results Table ──────────────────────────────\n")
    f.write(summary_df.to_latex(
        index=False, escape=True, column_format="llllrl",
        caption="IV and RDD Results Summary",
        label="tab:results", position="H",
    ))

print("[OUTPUT] Saved tables.tex")
print("\n[DONE] All four phases complete.")
print("Outputs: first_stage.pdf, rdd_plot.pdf, density_plot.pdf, tables.tex")
# Scrape complete
