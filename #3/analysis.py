"""
=============================================================
BANA290 — AI Lab Assignment 3
DiD Analysis: AI Training Subsidy on Regional Employment
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
from linearmodels.panel import PanelOLS

warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════
# PHASE 1 — SCRAPE
# ═════════════════════════════════════════════════════════════

# [Copilot Prompt]
# "Scrape the index page of the labor archive portal using
#  BeautifulSoup to extract all href links to individual
#  district brief pages. Then loop through each URL, scrape
#  its HTML table into a DataFrame, and store all DataFrames
#  in a list. Combine them with pd.concat() at the end."

INDEX_URL = "https://bana290-assignment3.netlify.app/"
HEADERS   = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

def scrape_index(url: str) -> list:
    """
    [Copilot Prompt]
    "Fetch the index page, find all <a> tags whose href
     contains '/briefs/', and return a deduplicated list
     of absolute URLs."
    """
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/briefs/" in href:
            if href.startswith("http"):
                links.append(href)
            else:
                links.append("https://bana290-assignment3.netlify.app" + href)

    links = list(dict.fromkeys(links))   # deduplicate, preserve order
    print(f"[SCRAPE] Found {len(links)} brief URLs on index page.")
    return links


def scrape_brief(url: str) -> pd.DataFrame:
    """
    [Copilot Prompt]
    "Fetch a single brief page, locate the <table> element,
     parse all header and data rows into a pandas DataFrame,
     and add a source_url column for traceability."
    """
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    table = soup.find("table")
    if table is None:
        raise ValueError(f"No table found at {url}")

    rows = []
    headers = []
    for i, tr in enumerate(table.find_all("tr")):
        cells = [td.get_text(separator=" ", strip=True)
                 for td in tr.find_all(["th", "td"])]
        if i == 0:
            headers = cells
        elif cells:
            rows.append(cells)

    max_w = max(len(r) for r in rows) if rows else 0
    rows  = [r + [""] * (max_w - len(r)) for r in rows]

    df = pd.DataFrame(rows, columns=headers[:max_w])
    df["source_url"] = url
    print(f"[SCRAPE]   {url.split('/')[-1]}: {len(df)} rows")
    return df


# ── Run the full scrape ──────────────────────────────────────
brief_urls = scrape_index(INDEX_URL)

frames = []
for url in brief_urls:
    try:
        frames.append(scrape_brief(url))
    except Exception as e:
        print(f"[SCRAPE] ERROR on {url}: {e}")

df_raw = pd.concat(frames, ignore_index=True)
print(f"[SCRAPE] Combined table: {len(df_raw)} rows × {len(df_raw.columns)} columns")
print(df_raw[["REGION", "PROGRAM_STATUS", "2018", "2022", "2025"]].to_string())


# ═════════════════════════════════════════════════════════════
# PHASE 2 — CLEAN
# ═════════════════════════════════════════════════════════════

df = df_raw.copy()

# ── 2a: Standardize region names ────────────────────────────

# [Copilot Prompt]
# "The REGION column contains bolded display names mixed with
#  sub-district labels (e.g. 'Lucas Cnty, Ohio Lake Erie
#  Production Belt'). Extract only the clean county name
#  before any newline or sub-label by splitting on newlines
#  and taking the first token."

def clean_region(text: str) -> str:
    """Keep only the first line of the region cell."""
    return str(text).split("\n")[0].strip()

df["REGION"] = df["REGION"].apply(clean_region)


# ── 2b: Map PROGRAM_STATUS to TREATED dummy ─────────────────

# [Copilot Prompt]
# "Create two keyword lists — one for treated labels (AI Grant
#  County, Grant Zone, Upskilling Cohort, Treated) and one for
#  control labels (Comparison Area, No Grant, Benchmark County,
#  Control). Write a function that returns 1 for treated,
#  0 for control, and raises a warning for ambiguous rows."

TREATED_KW = ["ai grant", "grant zone", "upskilling", "treated"]
CONTROL_KW = ["comparison", "no grant", "benchmark", "control"]

def map_treated(label: str) -> int:
    lo = str(label).lower().strip()
    for kw in TREATED_KW:
        if kw in lo:
            return 1
    for kw in CONTROL_KW:
        if kw in lo:
            return 0
    print(f"[CLEAN] Warning: ambiguous PROGRAM_STATUS '{label}' — defaulting to control")
    return 0

df["TREATED"] = df["PROGRAM_STATUS"].apply(map_treated)
print(f"\n[CLEAN] TREATED={df.TREATED.sum()} rows | CONTROL={(df.TREATED==0).sum()} rows")


# ── 2c: Reshape wide → long ──────────────────────────────────

# [Copilot Prompt]
# "The data is in wide format with year columns 2018–2025.
#  Use pd.melt() to reshape it so each row is a region-year
#  observation with columns REGION, YEAR, EMPLOYMENT,
#  TREATED, and STATE_GROUP. Drop the PORTAL_NOTE and
#  source_url columns before melting."

YEAR_COLS = [str(y) for y in range(2018, 2026)]
ID_COLS   = ["REGION", "STATE_GROUP", "TREATED"]

df_long = df[ID_COLS + YEAR_COLS].melt(
    id_vars    = ID_COLS,
    value_vars = YEAR_COLS,
    var_name   = "YEAR",
    value_name = "EMPLOYMENT_RAW",
)
df_long["YEAR"] = df_long["YEAR"].astype(int)
print(f"[CLEAN] After melt: {len(df_long)} rows")


# ── 2d: Convert "15.2k" employment strings to integers ───────

# [Copilot Prompt]
# "Employment figures use mixed formats: '32,620', '32,055 jobs',
#  '~30.9k', '31.4 thousand', '36,645', '34.6 K'.
#  Write a function that strips commas, removes non-numeric
#  words except 'k' and 'thousand', handles the ~ approximation
#  marker, multiplies by 1000 for k/thousand suffixes, and
#  returns a rounded integer."

def parse_employment(text: str) -> float:
    """
    [Copilot Prompt]
    "Use regex to extract the numeric part and detect k/thousand
     suffixes. Multiply accordingly and return an integer."
    """
    s = str(text).lower().strip()
    s = s.replace(",", "").replace("~", "").replace("jobs", "")
    s = s.replace("thousand", "k").strip()

    # Match number + optional k suffix
    m = re.search(r"([\d.]+)\s*k", s)
    if m:
        return round(float(m.group(1)) * 1000)

    m = re.search(r"[\d.]+", s)
    if m:
        return round(float(m.group(0)))

    return np.nan


df_long["EMPLOYMENT"] = df_long["EMPLOYMENT_RAW"].apply(parse_employment)
n_null = df_long["EMPLOYMENT"].isna().sum()
print(f"[CLEAN] Parsed employment. Nulls: {n_null}")
print(df_long.groupby(["REGION"])["EMPLOYMENT"].mean().round(0).to_string())


# ── 2e: Add POST_POLICY dummy and final cleaning ─────────────

# [Copilot Prompt]
# "Add a POST_POLICY dummy that equals 1 for years >= 2022
#  (the policy start year) and 0 otherwise. Also add a
#  DID_INTERACTION column = TREATED * POST_POLICY.
#  Drop any rows with missing EMPLOYMENT values."

POLICY_YEAR = 2022

df_long["POST_POLICY"]      = (df_long["YEAR"] >= POLICY_YEAR).astype(int)
df_long["DID_INTERACTION"]  = df_long["TREATED"] * df_long["POST_POLICY"]

df_long.dropna(subset=["EMPLOYMENT"], inplace=True)
df_long.reset_index(drop=True, inplace=True)

print(f"\n[CLEAN] Final panel: {len(df_long)} observations")
print(df_long[["REGION","YEAR","EMPLOYMENT","TREATED",
               "POST_POLICY","DID_INTERACTION"]].head(10).to_string())


# ═════════════════════════════════════════════════════════════
# PHASE 3 — ANALYZE
# ═════════════════════════════════════════════════════════════

sns.set_theme(style="whitegrid", font_scale=1.1)
YEARS = sorted(df_long["YEAR"].unique())


# ── 3a: Employment Trends Plot ───────────────────────────────

# [Copilot Prompt]
# "Compute the average employment by year and treatment group.
#  Plot two lines — Treated (Ohio) and Control (Pennsylvania)
#  — with a vertical dashed line at 2022 to mark the policy
#  start. Save as trends.pdf."

trend = (df_long.groupby(["YEAR", "TREATED"])["EMPLOYMENT"]
         .mean().reset_index())
trend["Group"] = trend["TREATED"].map({1: "Treated (Ohio)", 0: "Control (Pennsylvania)"})

fig, ax = plt.subplots(figsize=(10, 5))
for grp, color in [("Treated (Ohio)", "#DD8452"),
                   ("Control (Pennsylvania)", "#4C72B0")]:
    sub = trend[trend["Group"] == grp]
    ax.plot(sub["YEAR"], sub["EMPLOYMENT"], marker="o",
            label=grp, color=color, linewidth=2.2)

ax.axvline(POLICY_YEAR, color="crimson", linestyle="--",
           linewidth=1.5, label="Policy Start (2022)")
ax.set_title("Employment Trends: Treated vs Control Regions",
             fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Avg Employment")
ax.legend()
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig("trends.pdf", bbox_inches="tight", dpi=300)
plt.savefig("trends.png", bbox_inches="tight", dpi=150)
plt.close()
print("\n[OUTPUT] Saved trends.pdf / trends.png")


# ── 3b: Parallel Trends Test ────────────────────────────────

# [Copilot Prompt]
# "To test the parallel trends assumption, restrict data to
#  the pre-treatment period (before 2022). Run a regression
#  of EMPLOYMENT on YEAR * TREATED for pre-period only.
#  Plot the pre-period trends side by side and annotate with
#  the interaction p-value. Save as parallel_trends.pdf."

pre = df_long[df_long["YEAR"] < POLICY_YEAR].copy()
pre["YEAR_C"] = pre["YEAR"] - pre["YEAR"].min()

# OLS on pre-period: employment ~ year + treated + year*treated
from scipy.stats import linregress

pt_results = []
for grp, sub in pre.groupby("TREATED"):
    slope, intercept, r, p, se = linregress(sub["YEAR_C"], sub["EMPLOYMENT"])
    pt_results.append({"Group": grp, "Slope": round(slope, 1), "p": round(p, 4)})

pt_df = pd.DataFrame(pt_results)
print("\n── Parallel Trends (Pre-Period Slopes) ──────────────────")
print(pt_df.to_string(index=False))

# Visual parallel trends
fig, ax = plt.subplots(figsize=(9, 5))
pre_trend = pre.groupby(["YEAR","TREATED"])["EMPLOYMENT"].mean().reset_index()
for grp, color, label in [(1, "#DD8452", "Treated (Ohio)"),
                           (0, "#4C72B0", "Control (Pennsylvania)")]:
    sub = pre_trend[pre_trend["TREATED"] == grp]
    ax.plot(sub["YEAR"], sub["EMPLOYMENT"], marker="o",
            color=color, label=label, linewidth=2.2)

ax.set_title("Parallel Trends Check: Pre-Treatment Period (2018–2021)",
             fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Avg Employment")
ax.legend()
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig("parallel_trends.pdf", bbox_inches="tight", dpi=300)
plt.savefig("parallel_trends.png", bbox_inches="tight", dpi=150)
plt.close()
print("[OUTPUT] Saved parallel_trends.pdf / parallel_trends.png")


# ── 3c: Placebo Test ────────────────────────────────────────

# [Copilot Prompt]
# "Run a placebo DiD using 2020 as a fake policy year on
#  pre-treatment data only (2018-2021). If the placebo
#  interaction term is not significant, the parallel trends
#  assumption is further supported."

placebo_year = 2020
pre_placebo = pre.copy()
pre_placebo["POST_PLACEBO"] = (pre_placebo["YEAR"] >= placebo_year).astype(int)
pre_placebo["PLACEBO_DID"]  = pre_placebo["TREATED"] * pre_placebo["POST_PLACEBO"]

import statsmodels.formula.api as smf

placebo_model = smf.ols(
    "EMPLOYMENT ~ TREATED + POST_PLACEBO + PLACEBO_DID",
    data=pre_placebo
).fit()

placebo_coef = placebo_model.params["PLACEBO_DID"]
placebo_p    = placebo_model.pvalues["PLACEBO_DID"]

print(f"\n── Placebo Test (Fake Policy Year = {placebo_year}) ─────────")
print(f"  Placebo DiD coefficient : {placebo_coef:.2f}")
print(f"  p-value                 : {placebo_p:.4f}")
print(f"  Conclusion              : {'PASS — no anticipatory effect' if placebo_p > 0.1 else 'FAIL — potential anticipatory effect'}")


# ── 3d: DiD Fixed Effects Regression ────────────────────────

# [Copilot Prompt]
# "Run a two-way fixed effects DiD regression using
#  linearmodels PanelOLS with entity (county) fixed effects
#  and time fixed effects. The dependent variable is
#  EMPLOYMENT and the key regressor is DID_INTERACTION
#  (TREATED * POST_POLICY). Use clustered standard errors
#  at the entity level."

df_panel = df_long.copy()
df_panel = df_panel.set_index(["REGION", "YEAR"])

model = PanelOLS(
    dependent   = df_panel["EMPLOYMENT"],
    exog        = df_panel[["DID_INTERACTION"]],
    entity_effects = True,
    time_effects   = True,
)
result = model.fit(cov_type="clustered", cluster_entity=True)

did_coef = result.params["DID_INTERACTION"]
did_se   = result.std_errors["DID_INTERACTION"]
did_p    = result.pvalues["DID_INTERACTION"]
did_ci   = result.conf_int().loc["DID_INTERACTION"]

print("\n── DiD Fixed Effects Regression ──────────────────────────")
print(f"  DID Coefficient (ATE) : {did_coef:.2f}")
print(f"  Std Error             : {did_se:.2f}")
print(f"  p-value               : {did_p:.4f}")
print(f"  95% CI                : [{did_ci.iloc[0]:.2f}, {did_ci.iloc[1]:.2f}]")
print(f"\n{result.summary}")


# ── 3e: Export tables for LaTeX ──────────────────────────────

# [Copilot Prompt]
# "Export the DiD regression summary and placebo results
#  as a formatted LaTeX table. Also save the parallel trends
#  slope comparison as a small LaTeX table."

did_table = pd.DataFrame({
    "Coefficient":  [round(did_coef, 2)],
    "Std Error":    [round(did_se, 2)],
    "p-value":      [round(did_p, 4)],
    "95% CI Lower": [round(did_ci.iloc[0], 2)],
    "95% CI Upper": [round(did_ci.iloc[1], 2)],
}, index=["DID\\_INTERACTION (Treated × Post)"])

placebo_table = pd.DataFrame({
    "Fake Year": [placebo_year],
    "Coefficient": [round(placebo_coef, 2)],
    "p-value": [round(placebo_p, 4)],
    "Result": ["PASS" if placebo_p > 0.1 else "FAIL"],
})

with open("tables.tex", "w") as f:
    f.write("% ── DiD Regression Table ───────────────────────────\n")
    f.write(did_table.to_latex(
        escape=False, column_format="lrrrrr",
        caption="Two-Way Fixed Effects DiD Regression Results",
        label="tab:did", position="H",
    ))
    f.write("\n\n% ── Placebo Test Table ─────────────────────────────\n")
    f.write(placebo_table.to_latex(
        index=False, escape=True, column_format="rrrr",
        caption="Placebo DiD Test (Fake Policy Year = 2020)",
        label="tab:placebo", position="H",
    ))
    f.write("\n\n% ── Parallel Trends Slopes ─────────────────────────\n")
    f.write(pt_df.to_latex(
        index=False, escape=True, column_format="lrr",
        caption="Pre-Treatment Employment Slopes by Group",
        label="tab:pt", position="H",
    ))

print("\n[OUTPUT] Saved tables.tex")
print("[DONE] All four phases complete.")
print("Outputs: trends.pdf, parallel_trends.pdf, tables.tex")
# Scrape complete
# Clean complete
# Analyze complete
