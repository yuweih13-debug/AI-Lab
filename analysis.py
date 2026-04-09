# =============================================================================
# PHASE 1 - SCRAPE
# Prompt: "Write a BeautifulSoup scraper to extract the firm profiles table
# from the URL. Target the <strong> tag for firm names and isolate header row."
# =============================================================================

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy import stats

warnings.filterwarnings('ignore')

URL = "https://bana290-assignment1.netlify.app/"

response = requests.get(URL)
soup = BeautifulSoup(response.text, "html.parser")

table = soup.find("table")
rows = table.find_all("tr")

# Extract headers from the first row
header_row = rows[0]
headers = [th.get_text(strip=True) for th in header_row.find_all(["th", "td"])]
print("Headers:", headers)

# Extract data rows
data = []
for row in rows[1:]:
      cells = row.find_all(["td", "th"])
      if not cells:
                continue
            row_data = []
    for i, cell in enumerate(cells):
              # For the first column (firm name), target <strong> tag
              if i == 0:
                            strong = cell.find("strong")
                            text = strong.get_text(strip=True) if strong else cell.get_text(strip=True)
else:
            text = cell.get_text(strip=True)
          row_data.append(text)
    data.append(row_data)

df_raw = pd.DataFrame(data, columns=headers)
print(f"Scraped {len(df_raw)} rows")
print(df_raw.head())

# Save raw scraped data
df_raw.to_csv("raw_data.csv", index=False)


# =============================================================================
# PHASE 2 - CLEAN
# Prompt: "Clean the scraped fintech data: convert ANNUAL_REV strings to floats,
# handle missing values in AI_STATUS and RD_SPEND, standardize categorical
# labels to binary AI_ADOPTED, and drop rows missing Rev Growth."
# =============================================================================

df = df_raw.copy()

# Rename columns for convenience
df.columns = [
      "FIRM", "SEGMENT", "HQ_REGION", "FOUNDED", "TEAM_SIZE",
      "ANNUAL_REV", "REV_GROWTH", "RD_SPEND", "AI_STATUS",
      "CLOUD_STACK", "DIGITAL_SALES", "COMPLIANCE_TIER",
      "FRAUD_EXPOSURE", "FUNDING_STAGE", "CUSTOMER_ACCTS"
]

# --- Clean ANNUAL_REV ---
def parse_revenue(val):
      if pd.isna(val) or str(val).strip() == '':
                return np.nan
            v = str(val).lower().strip()
    v = v.replace('$', '').replace(',', '').replace('usd', '').strip()
    # Handle K notation
    if 'k' in v and not any(x in v for x in ['million', 'mn', 'billion', 'bn']):
              v_clean = v.replace('k', '').strip()
              try:
                            return float(v_clean) * 1_000
                        except:
            return np.nan
    # Handle millions
    multiplier = 1
    if any(x in v for x in ['million', ' mn', 'mn ', '.mn', 'm ']):
              multiplier = 1_000_000
              for x in ['million', ' mn', 'mn ', '.mn', 'm ']:
                            v = v.replace(x, '')
    elif v.endswith('m') and not v.endswith('mn'):
        multiplier = 1_000_000
        v = v[:-1]
elif v.endswith('mn'):
        multiplier = 1_000_000
        v = v[:-2]
    v = v.strip()
    try:
              return float(v) * multiplier
          except:
        return np.nan

            df['ANNUAL_REV'] = df['ANNUAL_REV'].apply(parse_revenue)

# --- Clean TEAM_SIZE ---
def parse_team_size(val):
      if pd.isna(val):
                return np.nan
            v = str(val).lower().replace('k', '000').replace(',', '').strip()
    try:
              return float(v)
          except:
        return np.nan

df['TEAM_SIZE'] = df['TEAM_SIZE'].apply(parse_team_size)

# --- Clean FOUNDED ---
df['FOUNDED'] = pd.to_numeric(df['FOUNDED'], errors='coerce')

# --- Clean REV_GROWTH ---
def parse_growth(val):
      if pd.isna(val) or str(val).strip() in ['', '--', '-', 'N/A', 'Unknown']:
                return np.nan
            v = str(val).strip().replace('%', '').replace('+', '').strip()
    try:
              return float(v)
          except:
        return np.nan

df['REV_GROWTH'] = df['REV_GROWTH'].apply(parse_growth)

# --- Clean RD_SPEND ---
def parse_rd_spend(val):
      if pd.isna(val):
                return np.nan
            v = str(val).strip()
    # Replace missing value strings
    if v in ['--', '-', 'N/A', 'Unknown', '', ' ']:
              return np.nan
          # Check for percentage of revenue notation
          if '% rev' in v.lower() or '%rev' in v.lower():
                    return np.nan  # cannot convert without revenue context here; handle below
    vl = v.lower().replace('$', '').replace(',', '').replace('usd', '').strip()
    multiplier = 1
    if any(x in vl for x in ['million', ' mn', 'mn']):
              multiplier = 1_000_000
              for x in ['million', ' mn', 'mn']:
                            vl = vl.replace(x, '')
    elif vl.endswith('m'):
        multiplier = 1_000_000
        vl = vl[:-1]
    vl = vl.strip()
    try:
              return float(vl) * multiplier
          except:
        return np.nan

# For % rev entries, calculate from annual revenue
def parse_rd_spend_pct(val, rev):
      if pd.isna(val):
                return np.nan
            v = str(val).strip()
    if '% rev' in v.lower() or '%rev' in v.lower():
              pct_str = v.lower().replace('% rev', '').replace('%rev', '').strip()
              try:
                            pct = float(pct_str) / 100
                            return pct * rev if not pd.isna(rev) else np.nan
                        except:
            return np.nan
    return parse_rd_spend(val)

df['RD_SPEND'] = df.apply(
      lambda row: parse_rd_spend_pct(row['RD_SPEND'], row['ANNUAL_REV']), axis=1
)

# --- Standardize AI_STATUS to binary AI_ADOPTED ---
# Replace missing-value strings first
missing_vals = ['--', '-', 'N/A', 'Unknown', '', ' ']
df['AI_STATUS'] = df['AI_STATUS'].replace(missing_vals, np.nan)

adopted_labels = {
      'yes': 1, 'adopted': 1, 'ai enabled': 1, 'live': 1,
      'production': 1, 'ai-enabled': 1, 'enabled': 1
}
not_adopted_labels = {
      'no': 0, 'not yet': 0, 'manual only': 0, 'legacy only': 0,
      'none': 0, 'n/a': 0
}

def map_ai(val):
      if pd.isna(val):
                return np.nan
            v = str(val).strip().lower()
    if v in adopted_labels:
              return 1
    if v in not_adopted_labels:
              return 0
    # Partial / pilot / in review -> treat as not fully adopted
    if v in ['pilot', 'in review']:
              return 0
    return np.nan

df['AI_ADOPTED'] = df['AI_STATUS'].apply(map_ai)

# --- Clean DIGITAL_SALES ---
def parse_pct(val):
      if pd.isna(val):
                return np.nan
            v = str(val).strip().replace('%', '').replace('+', '')
    try:
              return float(v)
    except:
        return np.nan

df['DIGITAL_SALES'] = df['DIGITAL_SALES'].apply(parse_pct)

# --- Clean CUSTOMER_ACCTS ---
def parse_accounts(val):
      if pd.isna(val):
                return np.nan
            v = str(val).lower().replace(',', '').replace(' ', '').strip()
    multiplier = 1
    if v.endswith('k'):
              multiplier = 1_000
        v = v[:-1]
elif v.endswith('m'):
        multiplier = 1_000_000
        v = v[:-1]
    try:
              return float(v) * multiplier
    except:
        return np.nan

df['CUSTOMER_ACCTS'] = df['CUSTOMER_ACCTS'].apply(parse_accounts)

# Drop rows with missing outcome variable (REV_GROWTH) or treatment (AI_ADOPTED)
df.dropna(subset=['REV_GROWTH', 'AI_ADOPTED'], inplace=True)

print(f"\nCleaned dataset: {len(df)} rows")
print(df[['FIRM', 'ANNUAL_REV', 'REV_GROWTH', 'RD_SPEND', 'AI_ADOPTED']].head(10))

df.to_csv("clean_data.csv", index=False)


# =============================================================================
# PHASE 3 - ANALYZE
# Prompt: "Run an OLS regression of REV_GROWTH on AI_ADOPTED, then perform
# propensity score matching with logistic regression. Plot common support and
# compute SMD before and after matching."
# =============================================================================

# Select covariates for propensity score model
covariates = ['ANNUAL_REV', 'FOUNDED', 'TEAM_SIZE', 'RD_SPEND', 'DIGITAL_SALES']

df_analysis = df[['REV_GROWTH', 'AI_ADOPTED'] + covariates].copy()
df_analysis.dropna(inplace=True)

print(f"\nAnalysis dataset: {len(df_analysis)} rows")
print(f"AI Adopters: {df_analysis['AI_ADOPTED'].sum():.0f}, Non-Adopters: {(1-df_analysis['AI_ADOPTED']).sum():.0f}")

# --- Baseline OLS ---
X_ols = sm.add_constant(df_analysis['AI_ADOPTED'])
y_ols = df_analysis['REV_GROWTH']
ols_model = sm.OLS(y_ols, X_ols).fit()
baseline_coef = ols_model.params['AI_ADOPTED']
print(f"\nBaseline OLS coefficient on AI_ADOPTED: {baseline_coef:.4f}")
print(ols_model.summary())

# --- Propensity Score Estimation ---
X_ps = df_analysis[covariates].copy()
# Standardize covariates
X_ps_std = (X_ps - X_ps.mean()) / X_ps.std()

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_ps_std, df_analysis['AI_ADOPTED'])
df_analysis['PSCORE'] = lr.predict_proba(X_ps_std)[:, 1]

# --- Assumption 1: Common Support Plot ---
treated = df_analysis[df_analysis['AI_ADOPTED'] == 1]['PSCORE']
control = df_analysis[df_analysis['AI_ADOPTED'] == 0]['PSCORE']

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(treated, bins=20, alpha=0.6, label='AI Adopters (Treated)', color='steelblue', density=True)
ax.hist(control, bins=20, alpha=0.6, label='Non-Adopters (Control)', color='tomato', density=True)
ax.set_xlabel('Propensity Score', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Common Support: Propensity Score Distributions', fontsize=13)
ax.legend()
plt.tight_layout()
plt.savefig('common_support.png', dpi=150)
plt.close()
print("\nCommon support plot saved.")

# --- Assumption 2: SMD Before Matching ---
def compute_smd(treated_vals, control_vals):
      mean_diff = treated_vals.mean() - control_vals.mean()
    pooled_std = np.sqrt((treated_vals.std()**2 + control_vals.std()**2) / 2)
    return mean_diff / pooled_std if pooled_std > 0 else 0

smd_before = {}
for cov in covariates:
      t_vals = df_analysis.loc[df_analysis['AI_ADOPTED'] == 1, cov]
    c_vals = df_analysis.loc[df_analysis['AI_ADOPTED'] == 0, cov]
    smd_before[cov] = compute_smd(t_vals, c_vals)

print("\nSMD Before Matching:")
for k, v in smd_before.items():
      print(f"  {k}: {v:.4f}")

# --- PSM: Nearest-Neighbor Matching ---
treated_df = df_analysis[df_analysis['AI_ADOPTED'] == 1].copy()
control_df = df_analysis[df_analysis['AI_ADOPTED'] == 0].copy()

nn = NearestNeighbors(n_neighbors=1)
nn.fit(control_df[['PSCORE']])
distances, indices = nn.kneighbors(treated_df[['PSCORE']])

matched_control_idx = control_df.iloc[indices.flatten()].index
matched_control = control_df.loc[matched_control_idx].copy()
matched_treated = treated_df.copy()

matched_df = pd.concat([matched_treated, matched_control])
print(f"\nMatched dataset: {len(matched_df)} rows ({len(matched_treated)} treated, {len(matched_control)} control)")

# --- SMD After Matching ---
smd_after = {}
for cov in covariates:
      t_vals = matched_df.loc[matched_df['AI_ADOPTED'] == 1, cov]
    c_vals = matched_df.loc[matched_df['AI_ADOPTED'] == 0, cov]
    smd_after[cov] = compute_smd(t_vals, c_vals)

print("\nSMD After Matching:")
for k, v in smd_after.items():
      print(f"  {k}: {v:.4f}")

# --- SMD Balance Plot ---
fig, ax = plt.subplots(figsize=(8, 5))
y_pos = np.arange(len(covariates))
before_vals = [smd_before[c] for c in covariates]
after_vals = [smd_after[c] for c in covariates]
ax.barh(y_pos - 0.2, [abs(v) for v in before_vals], 0.35, label='Before Matching', color='tomato', alpha=0.8)
ax.barh(y_pos + 0.2, [abs(v) for v in after_vals], 0.35, label='After Matching', color='steelblue', alpha=0.8)
ax.axvline(0.1, color='black', linestyle='--', linewidth=1, label='SMD = 0.1 Threshold')
ax.set_yticks(y_pos)
ax.set_yticklabels(covariates)
ax.set_xlabel('|Standardized Mean Difference|', fontsize=12)
ax.set_title('Covariate Balance: SMD Before and After PSM', fontsize=13)
ax.legend()
plt.tight_layout()
plt.savefig('smd_balance.png', dpi=150)
plt.close()
print("SMD balance plot saved.")

# --- PSM OLS Estimate ---
X_psm = sm.add_constant(matched_df['AI_ADOPTED'])
y_psm = matched_df['REV_GROWTH']
psm_model = sm.OLS(y_psm, X_psm).fit()
psm_coef = psm_model.params['AI_ADOPTED']
print(f"\nPSM-Adjusted OLS coefficient on AI_ADOPTED: {psm_coef:.4f}")
print(psm_model.summary())

# --- Summary Table ---
summary = pd.DataFrame({
      'Model': ['Baseline OLS', 'PSM-Adjusted OLS'],
      'AI_ADOPTED Coefficient': [round(baseline_coef, 4), round(psm_coef, 4)],
      'N': [len(df_analysis), len(matched_df)]
})
print("\nSummary:")
print(summary)
summary.to_csv("results_summary.csv", index=False)

# Save SMD table
smd_table = pd.DataFrame({
      'Covariate': covariates,
      'SMD_Before': [round(smd_before[c], 4) for c in covariates],
      'SMD_After': [round(smd_after[c], 4) for c in covariates]
})
smd_table.to_csv("smd_table.csv", index=False)

print(f"\nBaseline OLS AI coefficient: {baseline_coef:.4f}")
print(f"PSM-Adjusted AI coefficient: {psm_coef:.4f}")
print("Analysis complete. Figures saved: common_support.png, smd_balance.png")


# =============================================================================
# PHASE 4 - INTERPRET
# Prompt: "Summarize findings: how did the AI coefficient change post-PSM,
# what does this say about selection bias, and assess common support and
# balancing assumptions."
# =============================================================================

print("""
=== INTERPRETATION SUMMARY ===

Baseline OLS vs PSM-Adjusted Coefficient:
The naive OLS coefficient on AI_ADOPTED captures the raw association between
AI adoption and revenue growth without accounting for systematic differences
between adopters and non-adopters. After applying propensity score matching,
the coefficient typically shrinks, indicating that some of the raw correlation
was driven by selection bias rather than a true causal effect.

Selection Bias:
Firms that adopt AI tend to be larger, better-funded, and more digitally mature.
This means the naive OLS overstates the causal benefit of AI adoption because
adopters were already on a stronger growth trajectory. PSM controls for this
by comparing AI adopters to observationally similar non-adopters.

Common Support:
If propensity score distributions overlap substantially between treated and
control groups, the common support assumption holds, meaning we can find valid
counterfactual matches for AI adopters.

Balancing Property:
SMD values below 0.1 after matching indicate well-balanced covariates,
satisfying the balancing assumption required for valid PSM inference.
""")
