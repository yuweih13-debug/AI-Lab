# ============================================================
# BANA290 - Assignment 1: AI Lab
# North American Fintech & Financial Services Directory
# ============================================================

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

# ============================================================
# PHASE 1: SCRAPE
# Prompt: Scrape the HTML table from the fintech directory URL,
# extract firm names from <strong> tags, and build a DataFrame
# ============================================================

url = "https://bana290-assignment1.netlify.app/"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

table = soup.find("table")
rows = table.find_all("tr")

# Extract headers from first row
headers = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]

# Extract data rows, targeting <strong> tag for firm name
data = []
for row in rows[1:]:
    cells = row.find_all("td")
    if not cells:
        continue
    row_data = []
    for i, cell in enumerate(cells):
        if i == 0:
            strong = cell.find("strong")
            row_data.append(strong.get_text(strip=True) if strong else cell.get_text(strip=True))
        else:
            row_data.append(cell.get_text(strip=True))
    data.append(row_data)

df = pd.DataFrame(data, columns=headers)
print(f"Scraped {len(df)} rows")
print(df.head(3))


# ============================================================
# PHASE 2: CLEAN
# Prompt: Standardize all messy columns including ANNUAL_REV,
# REV_GROWTH, RD_SPEND, AI_STATUS, TEAM_SIZE, CUSTOMER_ACCTS
# ============================================================

# --- Helper: parse revenue/spend strings to float ---
def parse_money(val):
    if pd.isna(val):
        return np.nan
    s = str(val).lower().strip()
    s = s.replace("usd", "").replace("$", "").replace(",", "").strip()
    if s in ["--", "n/a", "", "unknown", "-"]:
        return np.nan
    try:
        if "billion" in s or "bn" in s:
            s = s.replace("billion", "").replace("bn", "").strip()
            return float(s) * 1_000_000_000
        elif "million" in s or "mn" in s or s.endswith("m"):
            s = s.replace("million", "").replace("mn", "").replace("m", "").strip()
            return float(s) * 1_000_000
        elif "k" in s:
            s = s.replace("k", "").strip()
            return float(s) * 1_000
        else:
            return float(s)
    except:
        return np.nan

# --- Helper: parse percentage strings to float ---
def parse_pct(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s in ["--", "N/A", "", "Unknown", "-"]:
        return np.nan
    s = s.replace("+", "").replace("%", "").strip()
    try:
        return float(s)
    except:
        return np.nan

# --- Clean ANNUAL_REV ---
df["ANNUAL_REV"] = df["Annual Rev."].apply(parse_money)

# --- Clean REV_GROWTH ---
df["REV_GROWTH"] = df["Rev Growth (YoY)"].apply(parse_pct)

# --- Clean TEAM_SIZE ---
def parse_team(val):
    if pd.isna(val):
        return np.nan
    s = str(val).lower().replace(",", "").strip()
    if "k" in s:
        return float(s.replace("k", "").strip()) * 1000
    try:
        return float(s)
    except:
        return np.nan

df["TEAM_SIZE"] = df["Team Size"].apply(parse_team)

# --- Clean CUSTOMER_ACCTS ---
def parse_accts(val):
    if pd.isna(val):
        return np.nan
    s = str(val).lower().replace(",", "").strip()
    if "m" in s:
        return float(s.replace("m", "").strip()) * 1_000_000
    elif "k" in s:
        return float(s.replace("k", "").strip()) * 1_000
    try:
        return float(s)
    except:
        return np.nan

df["CUSTOMER_ACCTS"] = df["Customer Accts"].apply(parse_accts)

# --- Clean RD_SPEND (handle % rev notation separately) ---
def parse_rd(row):
    val = str(row["R&D Spend"]).strip()
    if val in ["--", "N/A", "", "Unknown", "-"]:
        return np.nan
    if "% rev" in val.lower() or "%rev" in val.lower():
        pct_str = val.lower().replace("% rev", "").replace("%rev", "").strip()
        try:
            pct = float(pct_str) / 100
            return pct * row["ANNUAL_REV"]
        except:
            return np.nan
    return parse_money(val)

df["RD_SPEND"] = df.apply(parse_rd, axis=1)

# --- Clean AI_STATUS to binary (AI_ADOPTED) ---
ai_positive = ["yes", "adopted", "ai enabled", "live", "production"]

def parse_ai(val):
    if pd.isna(val):
        return np.nan
    s = str(val).lower().strip()
    if s in ["--", "n/a", "", "unknown", "-"]:
        return np.nan
    return 1 if s in ai_positive else 0

df["AI_ADOPTED"] = df["AI Program"].apply(parse_ai)

# --- Drop rows missing critical outcome or treatment variable ---
df.dropna(subset=["REV_GROWTH", "AI_ADOPTED"], inplace=True)
print(f"\nAfter dropping incomplete rows: {len(df)} rows remain")
print(df[["Firm", "ANNUAL_REV", "REV_GROWTH", "RD_SPEND", "AI_ADOPTED", "TEAM_SIZE"]].head(5))


# ============================================================
# PHASE 3: ANALYZE
# Prompt: Run OLS baseline regression, then apply Propensity
# Score Matching (PSM) with logistic regression and nearest
# neighbor matching. Check common support and SMD balance.
# ============================================================

# --- 3a: Baseline OLS regression ---
ols_model = smf.ols("REV_GROWTH ~ AI_ADOPTED", data=df).fit()
baseline_coef = ols_model.params["AI_ADOPTED"]
print(f"\n--- Baseline OLS Coefficient for AI_ADOPTED: {baseline_coef:.4f} ---")
print(ols_model.summary())

# --- Prepare covariates for PSM (drop rows missing covariates) ---
covariates = ["ANNUAL_REV", "TEAM_SIZE", "RD_SPEND"]
df_psm = df.dropna(subset=covariates + ["AI_ADOPTED", "REV_GROWTH"]).copy()
df_psm = df_psm.reset_index(drop=True)

X = df_psm[covariates].values
y = df_psm["AI_ADOPTED"].values

# --- 3b: Estimate propensity scores via logistic regression ---
lr = LogisticRegression(max_iter=1000)
lr.fit(X, y)
df_psm["PROPENSITY_SCORE"] = lr.predict_proba(X)[:, 1]

# --- Common Support: Plot propensity score distributions ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df_psm.loc[df_psm["AI_ADOPTED"] == 1, "PROPENSITY_SCORE"],
        bins=20, alpha=0.6, label="AI Adopted (Treated)", color="steelblue")
ax.hist(df_psm.loc[df_psm["AI_ADOPTED"] == 0, "PROPENSITY_SCORE"],
        bins=20, alpha=0.6, label="No AI (Control)", color="tomato")
ax.set_xlabel("Propensity Score")
ax.set_ylabel("Count")
ax.set_title("Common Support: Propensity Score Distribution by Group")
ax.legend()
plt.tight_layout()
plt.savefig("propensity_scores.png", dpi=150)
plt.show()
print("Saved: propensity_scores.png")

# --- 3c: SMD before matching ---
def compute_smd(treated, control):
    mean_diff = treated.mean() - control.mean()
    pooled_std = np.sqrt((treated.std()**2 + control.std()**2) / 2)
    return mean_diff / pooled_std if pooled_std != 0 else 0

treated = df_psm[df_psm["AI_ADOPTED"] == 1]
control = df_psm[df_psm["AI_ADOPTED"] == 0]

print("\n--- SMD Before Matching ---")
for cov in covariates:
    smd = compute_smd(treated[cov], control[cov])
    print(f"  {cov}: SMD = {smd:.4f}")

# --- 3d: Nearest-Neighbor Matching (1:1 without replacement) ---
nn = NearestNeighbors(n_neighbors=1)
nn.fit(control[["PROPENSITY_SCORE"]])
distances, indices = nn.kneighbors(treated[["PROPENSITY_SCORE"]])

matched_control = control.iloc[indices.flatten()].copy()
matched_treated = treated.copy()

matched_df = pd.concat([matched_treated, matched_control], ignore_index=True)

# --- SMD after matching ---
mt = matched_df[matched_df["AI_ADOPTED"] == 1]
mc = matched_df[matched_df["AI_ADOPTED"] == 0]

print("\n--- SMD After Matching ---")
for cov in covariates:
    smd = compute_smd(mt[cov], mc[cov])
    print(f"  {cov}: SMD = {smd:.4f}")

# --- PSM OLS regression on matched sample ---
psm_model = smf.ols("REV_GROWTH ~ AI_ADOPTED", data=matched_df).fit()
psm_coef = psm_model.params["AI_ADOPTED"]
print(f"\n--- PSM OLS Coefficient for AI_ADOPTED: {psm_coef:.4f} ---")
print(psm_model.summary())

print(f"\n=== Summary ===")
print(f"  Baseline OLS Coefficient : {baseline_coef:.4f}")
print(f"  PSM OLS Coefficient      : {psm_coef:.4f}")
print(f"  Difference               : {baseline_coef - psm_coef:.4f}")


# ============================================================
# PHASE 4: INTERPRET
# See interpretation.tex for the written findings (~250 words)
# addressing: coefficient shift, selection bias, common support,
# and balancing assumption validity.
# ============================================================

print("\nAnalysis complete. See interpretation.tex for written findings.")
# Scrape complete
