#!/usr/bin/env python3
"""
Build manuscript from R analysis output (02_analysis.R).
Reads CSVs, generates: LaTeX table, figure, .tex, .pdf.
Nothing is hardcoded — all numbers come from the R output.
"""

import pandas as pd
import numpy as np
import os
import subprocess
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Paths ─────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA  = os.path.join(BASE, "data")
FIGS  = os.path.join(BASE, "figures")
os.makedirs(FIGS, exist_ok=True)

# ── Load R output ─────────────────────────────────────────────
print("Loading R output...")
prev_df   = pd.read_csv(os.path.join(DATA, "prevalences.csv"))
rr_df     = pd.read_csv(os.path.join(DATA, "risk_ratios.csv"))
race_rr   = pd.read_csv(os.path.join(DATA, "race_stratified_rr.csv"))
racism_p  = pd.read_csv(os.path.join(DATA, "racism_prevalence.csv"))

# Outcome order and metadata
OUTCOMES = [
    ("Persistent sadness or hopelessness", "Mental Health", "b",
     "Felt so sad or hopeless almost every day for $\\geq$2 weeks that usual activities were stopped (past 12 months)."),
    ("Seriously considered suicide", "Mental Health", "c",
     "Seriously considered attempting suicide (past 12 months)."),
    ("Made a suicide plan", "Mental Health", "d",
     "Made a plan about how to attempt suicide (past 12 months)."),
    ("Attempted suicide", "Mental Health", "e",
     "Attempted suicide $\\geq$1 time (past 12 months)."),
    ("Bullied at school", "Violence", "f",
     "Bullied on school property (past 12 months)."),
    ("Electronically bullied", "Violence", "g",
     "Electronically bullied (past 12 months)."),
    ("Physical fight at school", "Violence", "h",
     "In a physical fight on school property $\\geq$1 time (past 12 months)."),
    ("Carried weapon at school", "Violence", "i",
     "Carried a weapon on school property $\\geq$1 day (past 30 days)."),
    ("Current e-cigarette use", "Substance Use", "j",
     "Used an electronic vapor product $\\geq$1 day (past 30 days)."),
    ("Current alcohol use", "Substance Use", "k",
     "Had $\\geq$1 drink of alcohol $\\geq$1 day (past 30 days)."),
]

RACISM_LEVELS = ["Never", "Rarely", "Sometimes", "Most", "Always"]

OUTCOME_QN = {
    "Persistent sadness or hopelessness": "QN26",
    "Seriously considered suicide": "QN27",
    "Made a suicide plan": "QN28",
    "Attempted suicide": "QN29",
    "Bullied at school": "QN24",
    "Electronically bullied": "QN25",
    "Physical fight at school": "QN17",
    "Carried weapon at school": "QN12",
    "Current e-cigarette use": "QN36",
    "Current alcohol use": "QN42",
}


def fp(p, lo, hi):
    """Format prevalence with CI for table cell."""
    if pd.isna(p): return '---'
    return f'{p:.1f} ({lo:.1f}\u2013{hi:.1f})'

def fp_tex(p, lo, hi):
    """Format prevalence with CI for LaTeX table cell."""
    if pd.isna(p): return '---'
    return f'{p:.1f} ({lo:.1f}--{hi:.1f})'

def fr(rr, lo, hi, is_ref=False):
    """Format risk ratio."""
    if is_ref: return '1 [Reference]'
    if pd.isna(rr): return '---'
    return f'{rr:.2f} ({lo:.2f}-{hi:.2f})'

def get_prev(outcome, level):
    r = prev_df[(prev_df['outcome'] == outcome) & (prev_df['racism_level'] == level)]
    return r.iloc[0]['pct'] if len(r) else np.nan

def get_rr(outcome, level):
    r = rr_df[(rr_df['outcome'] == outcome) & (rr_df['racism_level'] == level)]
    if len(r) and not pd.isna(r.iloc[0]['rr']):
        row = r.iloc[0]
        return row['rr'], row['rr_lo'], row['rr_hi']
    return np.nan, np.nan, np.nan

def wprev_py(y, w):
    """Weighted prevalence with approximate CI (matches R wprev)."""
    ok = y.notna() & w.notna() & (w > 0)
    yy = y[ok].values.astype(float)
    ww = w[ok].values.astype(float)
    n = len(yy)
    if n < 5:
        return np.nan, np.nan, np.nan, 0
    p = np.sum(yy * ww) / np.sum(ww)
    se = np.sqrt(np.sum(((yy - p) * ww) ** 2)) / np.sum(ww)
    return (100 * p,
            100 * max(0, p - 1.96 * se),
            100 * min(1, p + 1.96 * se),
            n)


# ── Compute non-White stats from raw data ─────────────────────
print("Computing non-White statistics...")
raw = pd.read_csv(os.path.join(DATA, "yrbs2023_parsed.csv"))
raw = raw[raw['Q23'].notna() & raw['weight'].notna() & (raw['weight'] > 0)]

# Recode QN outcomes: 1=yes, 2=no → 1/0
for qn in OUTCOME_QN.values():
    raw[f'{qn}_bin'] = np.where(raw[qn] == 1, 1, np.where(raw[qn] == 2, 0, np.nan))

# Non-White subset
nw = raw[(raw['raceeth'].notna()) & (raw['raceeth'] != 5)].copy()
nw_n = len(nw)

# Non-White most/always rate
nw_total_w = nw['weight'].astype(float).sum()
nw_ma_w = nw[nw['Q23'].isin([4, 5])]['weight'].astype(float).sum()
nw_ma_p = 100 * nw_ma_w / nw_total_w

# Non-White prevalences by outcome × racism level (for Panel B)
nw_prev_rows = []
lev_map = {1: 'Never', 2: 'Rarely', 3: 'Sometimes', 4: 'Most', 5: 'Always'}
for label, domain, _, _ in OUTCOMES:
    qn = OUTCOME_QN[label]
    col = f'{qn}_bin'
    for lev_val, lev_name in lev_map.items():
        sub = nw[nw['Q23'] == lev_val]
        pct, lo, hi, n = wprev_py(sub[col], sub['weight'])
        nw_prev_rows.append({
            'outcome': label, 'racism_level': lev_name,
            'pct': pct, 'lo': lo, 'hi': hi, 'n': n
        })
nw_prev_df = pd.DataFrame(nw_prev_rows)

# 3-group prevalences for text (suicide ideation example)
sui_col = 'QN27_bin'
grp_never = raw[raw['Q23'] == 1]
grp_rs = raw[raw['Q23'].isin([2, 3])]
grp_ma = raw[raw['Q23'].isin([4, 5])]
sui_never, _, _, _ = wprev_py(grp_never[sui_col], grp_never['weight'])
sui_rs, _, _, _ = wprev_py(grp_rs[sui_col], grp_rs['weight'])
sui_ma, _, _, _ = wprev_py(grp_ma[sui_col], grp_ma['weight'])


# ── Pull key numbers for text ─────────────────────────────────
ov = racism_p[racism_p['group'] == 'Overall'].iloc[0]
n_valid = int(ov['n'])
any_p, any_lo, any_hi = ov['any_pct'], ov['any_lo'], ov['any_hi']

race_pcts = {}
for _, row in racism_p.iterrows():
    race_pcts[row['group']] = row['any_pct']

# Bullied at school RRs (for 3-group text: Rarely/Sometimes range, Most/Always range)
bul_rr_rarely, bul_rr_rarely_lo, bul_rr_rarely_hi = get_rr("Bullied at school", "Rarely")
bul_rr_sometimes, bul_rr_sometimes_lo, bul_rr_sometimes_hi = get_rr("Bullied at school", "Sometimes")
bul_rr_most, bul_rr_most_lo, bul_rr_most_hi = get_rr("Bullied at school", "Most")
bul_rr_alw, bul_rr_alw_lo, bul_rr_alw_hi = get_rr("Bullied at school", "Always")

# CI ranges for collapsed groups
bul_rs_ci_lo = min(bul_rr_rarely_lo, bul_rr_sometimes_lo)
bul_rs_ci_hi = max(bul_rr_rarely_hi, bul_rr_sometimes_hi)
bul_ma_ci_lo = min(bul_rr_most_lo, bul_rr_alw_lo)
bul_ma_ci_hi = max(bul_rr_most_hi, bul_rr_alw_hi)

# Race-stratified text (Asian, Black) — LaTeX version for manuscript
race_parts = []
for race in ['Asian', 'Black']:
    sub = race_rr[race_rr['race'] == race]
    parts_inner = []
    for _, row in sub.iterrows():
        parts_inner.append(
            f'``{row["racism_level"]}\'\' RR {row["rr"]:.2f}, '
            f'95\\% CI {row["rr_lo"]:.2f}--{row["rr_hi"]:.2f}')
    if parts_inner:
        race_parts.append(f'{race} children ({"; ".join(parts_inner)})')
race_text = ' and '.join(race_parts)

# Plain-text version for word count
race_parts_plain = []
for race in ['Asian', 'Black']:
    sub = race_rr[race_rr['race'] == race]
    parts_inner = []
    for _, row in sub.iterrows():
        parts_inner.append(
            f'\u201c{row["racism_level"]}\u201d RR {row["rr"]:.2f}, '
            f'95% CI {row["rr_lo"]:.2f}\u2013{row["rr_hi"]:.2f}')
    if parts_inner:
        race_parts_plain.append(f'{race} children ({"; ".join(parts_inner)})')
race_text_plain = ' and '.join(race_parts_plain)


# ── Generate LaTeX table (landscape, two panels, with CI) ──────
print("Generating LaTeX table...")

def make_panel_tabular(panel_df):
    """Build tabular body rows for one panel."""
    rows = []
    current_domain = None
    for label, domain, _, _ in OUTCOMES:
        if domain != current_domain:
            if current_domain is not None:
                rows.append(r'\addlinespace[4pt]')
            rows.append(f'\\textit{{{domain}}} \\\\')
            current_domain = domain

        # Total N = sum across all racism levels for this outcome
        total_n = int(panel_df[panel_df['outcome'] == label]['n'].sum())

        prev_cells = []
        for lev in RACISM_LEVELS:
            pr = panel_df[(panel_df['outcome'] == label) &
                          (panel_df['racism_level'] == lev)]
            if len(pr) and not pd.isna(pr.iloc[0]['pct']):
                p = pr.iloc[0]
                prev_cells.append(fp_tex(p['pct'], p['lo'], p['hi']))
            else:
                prev_cells.append('---')

        rows.append(f'\\quad {label} & {total_n:,} & '
                     + ' & '.join(prev_cells) + r' \\')
    return '\n'.join(rows)

panel_a = make_panel_tabular(prev_df)
panel_b = make_panel_tabular(nw_prev_df)

table_tex = rf"""\begin{{landscape}}
\begin{{table}}[p]
\centering
\caption{{Weighted Prevalence (\%) of Health Outcomes by Frequency of School-Based Racial Discrimination, 2023 YRBS}}
\label{{tab:main}}
\scriptsize

\textbf{{Panel A: Full Sample ($N = {n_valid:,}$)}}
\vspace{{4pt}}

\begin{{tabular}}{{l r ccccc}}
\toprule
 & & \multicolumn{{5}}{{c}}{{Weighted Prevalence, \% (95\% CI)}} \\
\cmidrule(lr){{3-7}}
Outcome & $N$ & Never & Rarely & Sometimes & Most of the time & Always \\
\midrule
{panel_a}
\bottomrule
\end{{tabular}}

\vspace{{12pt}}

\textbf{{Panel B: Non-White Students ($N = {nw_n:,}$)}}
\vspace{{4pt}}

\begin{{tabular}}{{l r ccccc}}
\toprule
 & & \multicolumn{{5}}{{c}}{{Weighted Prevalence, \% (95\% CI)}} \\
\cmidrule(lr){{3-7}}
Outcome & $N$ & Never & Rarely & Sometimes & Most of the time & Always \\
\midrule
{panel_b}
\bottomrule
\end{{tabular}}

\end{{table}}
\end{{landscape}}"""


# ── Generate figure (faceted ARR dose-response with CI ribbon bands) ──
print("Generating figure...")

DOMAIN_COLORS_FIG = {
    'Mental Health': '#2166ac',
    'Violence': '#b2182b',
    'Substance Use': '#e66101',
}

fig, axes = plt.subplots(2, 5, figsize=(16, 7))
axes = axes.flatten()

for idx, (label, domain, _, _) in enumerate(OUTCOMES):
    ax = axes[idx]
    x = list(range(5))
    y, lo, hi = [], [], []

    # Never = reference (RR=1)
    y.append(1.0); lo.append(1.0); hi.append(1.0)

    for lev in ["Rarely", "Sometimes", "Most", "Always"]:
        row = rr_df[(rr_df['outcome'] == label) & (rr_df['racism_level'] == lev)]
        if len(row) and not pd.isna(row.iloc[0]['rr']):
            r = row.iloc[0]
            y.append(r['rr']); lo.append(r['rr_lo']); hi.append(r['rr_hi'])
        else:
            y.append(np.nan); lo.append(np.nan); hi.append(np.nan)

    color = DOMAIN_COLORS_FIG.get(domain, '#333')
    ax.fill_between(x, lo, hi, alpha=0.25, color=color, linewidth=0)
    ax.plot(x, y, '-o', color=color, markersize=5, linewidth=2.0)
    ax.axhline(1.0, color='#888888', linewidth=0.7, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(['Never', 'Rarely', 'Sometimes', 'Most of\nthe time', 'Always'],
                       fontsize=9, rotation=45, ha='right')
    ax.set_yscale('log')
    ax.set_yticks([1, 1.5, 2, 3, 5, 8])
    ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis='y', labelsize=10)
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color('#333333')
    ax.grid(True, which='major', axis='both', color='#cccccc', linewidth=0.5, linestyle='-')
    ax.set_axisbelow(True)

    # Title wrapping
    short = label
    if len(short) > 22:
        words = short.split(); mid = len(words) // 2
        short = ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
    ax.set_title(short, fontsize=12, fontweight='bold', pad=8)
    if idx % 5 == 0:
        ax.set_ylabel('Adjusted\nRisk Ratio', fontsize=11)
    if idx >= 5:
        ax.set_xlabel('', fontsize=11)

plt.tight_layout()
fig.savefig(os.path.join(FIGS, 'figure1.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved figure")


# ── Build supplement eTable (as plain string to avoid rf-string escaping) ──
supplement_tex = r"""
\singlespacing
\small
\renewcommand{\arraystretch}{1.2}
\begin{longtable}{>{\raggedright\arraybackslash}p{1.6in} >{\raggedright\arraybackslash}p{2.6in} >{\raggedright\arraybackslash}p{2.0in}}
\toprule
\textbf{Variable} & \textbf{Survey Question} & \textbf{Response Options / Coding} \\
\midrule
\endfirsthead
\toprule
\textbf{Variable} & \textbf{Survey Question} & \textbf{Response Options / Coding} \\
\midrule
\endhead
\bottomrule
\endfoot

\multicolumn{3}{l}{\textbf{\textit{Exposure}}} \\
\midrule

School-based racism (Q23) &
``During your life, how often have you felt that you were treated badly or unfairly
in school because of your race or ethnicity?'' &
1 = Never; 2 = Rarely; 3 = Sometimes; 4 = Most of the time; 5 = Always.
Coded as ordinal (5 levels; reference = Never). \\

\midrule
\multicolumn{3}{l}{\textbf{\textit{Outcomes --- Mental Health}}} \\
\midrule

Persistent sadness (QN26) &
``During the past 12 months, did you ever feel so sad or hopeless almost every day for
two weeks or more in a row that you stopped doing some usual activities?'' &
Yes / No. Coded as binary. \\
\addlinespace[3pt]

Suicidal ideation (QN27) &
``During the past 12 months, did you ever seriously consider attempting suicide?'' &
Yes / No. Coded as binary. \\
\addlinespace[3pt]

Suicide plan (QN28) &
``During the past 12 months, did you make a plan about how you would attempt suicide?'' &
Yes / No. Coded as binary. \\
\addlinespace[3pt]

Suicide attempt (QN29) &
``During the past 12 months, how many times did you actually attempt suicide?'' &
0 times / 1 time / 2--3 / 4--5 / 6+.
Coded as binary (1 if $\geq$1 time). \\

\midrule
\multicolumn{3}{l}{\textbf{\textit{Outcomes --- Violence}}} \\
\midrule

Bullied at school (QN24) &
``During the past 12 months, have you ever been bullied on school property?'' &
Yes / No. Coded as binary. \\
\addlinespace[3pt]

Electronically bullied (QN25) &
``During the past 12 months, have you ever been electronically bullied (counting being
bullied through texting, Instagram, Facebook, or other social media)?'' &
Yes / No. Coded as binary. \\
\addlinespace[3pt]

Physical fight at school (QN17) &
``During the past 12 months, how many times were you in a physical fight on school
property?'' &
0 / 1 / 2--3 / 4--5 / 6--7 / 8--9 / 10--11 / 12+ times.
Coded as binary (1 if $\geq$1 time). \\
\addlinespace[3pt]

Carried weapon at school (QN12) &
``During the past 30 days, on how many days did you carry a weapon such as a gun,
knife, or club on school property?'' &
0 / 1 / 2--3 / 4--5 / 6+ days.
Coded as binary (1 if $\geq$1 day). \\

\midrule
\multicolumn{3}{l}{\textbf{\textit{Outcomes --- Substance Use}}} \\
\midrule

Current e-cigarette use (QN36) &
``During the past 30 days, on how many days did you use an electronic vapor product?'' &
0 / 1--2 / 3--5 / 6--9 / 10--19 / 20--29 / All 30 days.
Coded as binary (1 if $\geq$1 day). \\
\addlinespace[3pt]

Current alcohol use (QN42) &
``During the past 30 days, on how many days did you have at least one drink of
alcohol?'' &
0 / 1--2 / 3--5 / 6--9 / 10--19 / 20--29 / All 30 days.
Coded as binary (1 if $\geq$1 day). \\

\midrule
\multicolumn{3}{l}{\textbf{\textit{Covariates}}} \\
\midrule

Sex (Q2) &
``What is your sex?'' &
Female / Male. Coded as binary (1 = Female). \\
\addlinespace[3pt]

Age (Q1) &
``How old are you?'' &
12 years or younger / 13 / 14 / 15 / 16 / 17 / 18 or older.
Coded as categorical (7 levels). \\
\addlinespace[3pt]

Race/ethnicity (raceeth) &
Derived from Q4 (``Are you Hispanic or Latino?'') and Q5 (``What is your race?''). &
8 categories: AI/AN, Asian, Black, NH/PI, White, Hispanic, Multiple-Hispanic,
Multiple-Non-Hispanic.
Coded as categorical (8 levels). \\
\addlinespace[3pt]

Grades (QN87) &
``During the past 12 months, how would you describe your grades in school?'' &
Mostly A's / B's / C's / D's / F's / None / Not sure.
Coded as binary (1 if A's or B's). \\
\addlinespace[3pt]

English proficiency (QN107) &
``How well do you speak English?'' &
Very well / Well / Not well / Not at all.
Coded as binary (1 if Very well or Well). \\

\end{longtable}
\doublespacing
\normalsize
"""

# ── Compute word count (body only: Introduction through Discussion) ──
_body_parts = [
    # Introduction
    "Racism is one of the best documented and most powerful social determinants of "
    "health. However, there has been relatively little research studying the role "
    "that racism plays in the health of children. In this study, we examine the "
    "relationship between experiences with racism at school and important health behaviors "
    "among a nationally representative sample of US high school students.",
    # Methods
    f"This study uses data from the Centers for Disease Control and Prevention\u2019s 2023 "
    f"Youth Risk Behavior Surveillance System (YRBS; N = 20,103; overall response rate "
    f"35.4%), a biannual survey administered in US high schools on behavioral health "
    f"topics. The survey uses a three-stage cluster design to capture a representative "
    f"sample of high school students. The survey includes measures of mental health, substance "
    f"use, and violence victimization, among other important health behaviors (eTable 1 in the "
    f"Supplement). In 2023, the national YRBS introduced a measure of children\u2019s experience "
    f"with racism at school. Children were asked \u201cHow often have you felt that you were treated "
    f"badly or unfairly in school because of your race or ethnicity?\u201d with possible responses "
    f"\u201cNever,\u201d \u201cRarely,\u201d \u201cSometimes,\u201d \u201cMost of the time,\u201d and \u201cAlways.\u201d "
    f"First, the unadjusted prevalence of several risky behaviors is reported by level of "
    f"experience with racism (Table 1). Then, the relative risk of each of these behaviors was "
    f"estimated via logistic regression, accounting for sex, age, race/ethnicity, English "
    f"language proficiency, and grades, relative to those who experienced no racism. "
    f"These analyses were conducted collectively and by student race/ethnicity. Estimates are "
    f"presented as risk ratios, which are computed using random draws from the regression\u2019s "
    f"variance-covariance matrix and holding confounders at their means. "
    f"This study is exempt from institutional review as it only uses secondary analysis of "
    f"publicly available data.",
    # Results
    f"Of the {n_valid:,} YRBS participants with non-missing data, "
    f"{any_p:.1f}% (95% CI, {any_lo:.1f}%\u2013{any_hi:.1f}%) "
    f"reported experiencing some racism at school, including "
    f"{race_pcts.get('Asian', 0):.1f}% of Asian, "
    f"{race_pcts.get('Black', 0):.1f}% of Black, "
    f"{race_pcts.get('Multiple', 0):.1f}% of Multiracial, "
    f"{race_pcts.get('Hispanic', 0):.1f}% of Hispanic, and "
    f"{race_pcts.get('White', 0):.1f}% of White children. "
    f"{nw_ma_p:.1f}% of non-White children reported experiencing racism at school most of the "
    f"time or always. "
    f"Most negative health behaviors exhibit a dose-response relationship with children\u2019s "
    f"experience with racism (Table 1; Figure 1). For example, suicidal "
    f"ideation has a prevalence of {sui_never:.1f}% among children who report never "
    f"experiencing racism, {sui_rs:.1f}% among children who report experiencing racism rarely "
    f"or sometimes, and {sui_ma:.1f}% among children who report experiencing racism most of "
    f"the time or always. "
    f"These patterns persist after accounting for potential confounders. After controlling for "
    f"sex, age, race/ethnicity, English language proficiency, and grades, those who experience "
    f"racism rarely or sometimes (RR {bul_rr_rarely:.2f}\u2013{bul_rr_sometimes:.2f}, 95% CI "
    f"{bul_rs_ci_lo:.2f}\u2013{bul_rs_ci_hi:.2f}) and those who experience racism most of the time "
    f"or always (RR {bul_rr_most:.2f}\u2013{bul_rr_alw:.2f}, 95% CI "
    f"{bul_ma_ci_lo:.2f}\u2013{bul_ma_ci_hi:.2f}) are significantly more likely to be bullied at "
    f"school compared with those who never experience racism. The response is strongest for "
    f"{race_text_plain}.",
    # Discussion
    "A significant proportion of US high school children report experiencing racism in schools, "
    "and these experiences correlate significantly and in a dose-response relationship with "
    "several negative health outcomes. "
    "Racism is a well-recognized threat to public health. These results extend that "
    "finding to children in schools and merit a well-coordinated response from public "
    "health professionals. For example, policy-makers, educators, and clinicians may design "
    "interventions to reduce racism among young people and to screen children who experience "
    "racism at school for negative health behaviors. "
    "This study has several limitations. This cross-sectional study cannot establish causation. "
    "The YRBS is designed to be nationally representative, but its 35.4% response rate could "
    "limit generalizability. Though children respond privately and anonymously, self-report "
    "can be biased. The survey uses multiple choice responses, which precludes more detailed "
    "accounts of children\u2019s experience with racism. The survey did not capture student "
    "experiences with racism outside of school. "
    "Future research should seek a clearer understanding of how children experience racism "
    "at school, the causal effects of racism on child health, and which policies and "
    "interventions can reduce children\u2019s experience with racism.",
]
_body_text = ' '.join(_body_parts)
# Strip punctuation/numbers for word count
word_count = len(re.findall(r'[A-Za-z\u2019]+', _body_text))
print(f"  Body word count: {word_count}")

# ── Write LaTeX ───────────────────────────────────────────────
print("Writing LaTeX...")

tex = rf"""\documentclass[12pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{setspace}}
\usepackage{{caption}}
\usepackage{{float}}
\usepackage{{microtype}}
\usepackage{{longtable}}
\usepackage{{array}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\usepackage{{pdflscape}}
\usepackage[hidelinks]{{hyperref}}
\captionsetup{{font=normalsize, labelfont=bf}}
\setlength{{\parskip}}{{6pt plus 2pt minus 1pt}}
\doublespacing

\begin{{document}}
\raggedright

%% ── COVER PAGE ─────────────────────────────────────────────
\thispagestyle{{empty}}

\noindent\textbf{{RESEARCH LETTER}}

\vspace{{24pt}}

\noindent\textbf{{Racism in Schools and Health Among US Adolescents in the United States, 2023}}

\vspace{{24pt}}

\noindent Theodore L. Caputi\\
Massachusetts Institute of Technology, Department of Economics, Cambridge, MA

\vspace{{24pt}}

\noindent\textbf{{Author Affiliations:}} Massachusetts Institute of Technology, Department
of Economics, Cambridge, MA (Caputi).

\vspace{{12pt}}

\noindent\textbf{{Corresponding Author:}} Theodore L. Caputi, MIT Department of Economics,
50 Memorial Dr, Cambridge, MA 02142 (tcaputi@gmail.com, https://www.tlcaputi.com).

\vspace{{12pt}}

\noindent\textbf{{Conflict of Interest Disclosures:}} None reported.

\vspace{{12pt}}

\noindent\textbf{{Word Count:}} {word_count}

\clearpage

%% ── BODY ───────────────────────────────────────────────────
\noindent\textbf{{Introduction}}

Racism is one of the best documented and most powerful social determinants of
health.$^{{1,2}}$ However, there has been relatively little research studying the role
that racism plays in the health of children.$^{{3}}$ In this study, we examine the
relationship between experiences with racism at school and important health behaviors
among a nationally representative sample of US high school students.

\vspace{{6pt}}

\noindent\textbf{{Methods}}

This study uses data from the Centers for Disease Control and Prevention's 2023
Youth Risk Behavior Surveillance System (YRBS; $N = 20,103$; overall response rate
35.4\%), a biannual survey administered in US high schools on behavioral health
topics.$^{{4}}$ The survey uses a three-stage cluster design to capture a representative
sample of high school students. The survey includes measures of mental health, substance
use, and violence victimization, among other important health behaviors (eTable 1 in the
Supplement). In 2023, the national YRBS introduced a measure of children's experience
with racism at school. Children were asked ``How often have you felt that you were treated
badly or unfairly in school because of your race or ethnicity?'' with possible responses
``Never,'' ``Rarely,'' ``Sometimes,'' ``Most of the time,'' and ``Always.''

First, the unadjusted prevalence of several risky behaviors is reported by level of
experience with racism (\autoref{{tab:main}}). Then, the relative risk of each of these
behaviors was estimated via logistic regression, accounting for sex, age, race/ethnicity,
English language proficiency, and grades, relative to those who experienced no racism.
These analyses were conducted collectively and by student race/ethnicity. Estimates are
presented as risk ratios, which are computed using random draws from the regression's
variance-covariance matrix and holding confounders at their means.$^{{5}}$

This study is exempt from institutional review as it only uses secondary analysis of
publicly available data.

\vspace{{6pt}}

\noindent\textbf{{Results}}

Of the {n_valid:,} YRBS participants with non-missing data,
{any_p:.1f}\% (95\% CI, {any_lo:.1f}\%--{any_hi:.1f}\%)
reported experiencing some racism at school, including
{race_pcts.get('Asian', 0):.1f}\% of Asian,
{race_pcts.get('Black', 0):.1f}\% of Black,
{race_pcts.get('Multiple', 0):.1f}\% of Multiracial,
{race_pcts.get('Hispanic', 0):.1f}\% of Hispanic, and
{race_pcts.get('White', 0):.1f}\% of White children.
{nw_ma_p:.1f}\% of non-White children reported experiencing racism at school most of the
time or always.

Most negative health behaviors exhibit a dose-response relationship with children's
experience with racism (\autoref{{tab:main}}; \autoref{{fig:rr}}). For example, suicidal
ideation has a prevalence of {sui_never:.1f}\% among children who report never
experiencing racism, {sui_rs:.1f}\% among children who report experiencing racism rarely
or sometimes, and {sui_ma:.1f}\% among children who report experiencing racism most of
the time or always.

These patterns persist after accounting for potential confounders. After controlling for
sex, age, race/ethnicity, English language proficiency, and grades, those who experience
racism rarely or sometimes (RR {bul_rr_rarely:.2f}--{bul_rr_sometimes:.2f}, 95\% CI
{bul_rs_ci_lo:.2f}--{bul_rs_ci_hi:.2f}) and those who experience racism most of the time
or always (RR {bul_rr_most:.2f}--{bul_rr_alw:.2f}, 95\% CI
{bul_ma_ci_lo:.2f}--{bul_ma_ci_hi:.2f}) are significantly more likely to be bullied at
school compared with those who never experience racism. The response is strongest for
{race_text}.

\vspace{{6pt}}

\noindent\textbf{{Discussion}}

A significant proportion of US high school children report experiencing racism in schools,
and these experiences correlate significantly and in a dose-response relationship with
several negative health outcomes.

Racism is a well-recognized threat to public health.$^{{1,2}}$ These results extend that
finding to children in schools$^{{3}}$ and merit a well-coordinated response from public
health professionals. For example, policy-makers, educators, and clinicians may design
interventions to reduce racism among young people and to screen children who experience
racism at school for negative health behaviors.$^{{6}}$

This study has several limitations. This cross-sectional study cannot establish causation.
The YRBS is designed to be nationally representative, but its 35.4\% response rate could
limit generalizability. Though children respond privately and anonymously, self-report
can be biased. The survey uses multiple choice responses, which precludes more detailed
accounts of children's experience with racism. The survey did not capture student
experiences with racism outside of school.

Future research should seek a clearer understanding of how children experience racism
at school, the causal effects of racism on child health, and which policies and
interventions can reduce children's experience with racism.

%% ── REFERENCES ─────────────────────────────────────────────
\clearpage

\noindent\textbf{{References}}

\begin{{enumerate}}
\small
\item Williams DR, Mohammed SA. Racism and health I: pathways and scientific evidence.
  \textit{{Am Behav Sci}}. 2013;57(8):1152-1173.
\item Paradies Y, Ben J, Denson N, et al. Racism as a determinant of health:
  a systematic review and meta-analysis. \textit{{PLoS One}}. 2015;10(9):e0138511.
\item Priest N, Paradies Y, Trenerry B, et al. A systematic review of studies examining
  the relationship between reported racism and health and wellbeing for children and young
  people. \textit{{Soc Sci Med}}. 2013;95:115-127.
\item Centers for Disease Control and Prevention. Youth Risk Behavior Surveillance
  System (YRBS). https://www.cdc.gov/yrbs/data/index.html. Accessed 2025.
\item King G, Tomz M, Wittenberg J. Making the most of statistical analyses: improving
  interpretation and presentation. \textit{{Am J Pol Sci}}. 2000;44(2):347-361.
\item Trent M, Dooley DG, Doug\'{{\e}} J; Section on Adolescent Health; Council on Community
  Pediatrics. The impact of racism on child and adolescent health.
  \textit{{Pediatrics}}. 2019;144(2):e20191765.
\end{{enumerate}}

%% ── TABLE (landscape) ──────────────────────────────────────
{table_tex}

%% ── FIGURE (landscape) ─────────────────────────────────────
\begin{{landscape}}
\begin{{figure}}[H]
  \centering
  \includegraphics[width=\linewidth]{{figures/figure1.pdf}}
  \caption{{Adjusted risk ratios for health outcomes by frequency of school-based
  racism among US high school students, 2023 YRBS ($N = {n_valid:,}$).
  Risk ratios from logistic regression computed via simulation from the
  variance-covariance matrix (King et al, 2000), adjusting for sex, age,
  race/ethnicity, English language proficiency, and grades.
  Shaded bands indicate 95\% confidence intervals. Reference category is Never.}}
  \label{{fig:rr}}
\end{{figure}}
\end{{landscape}}

%% ── SUPPLEMENT ──────────────────────────────────────────────
\clearpage
\setcounter{{page}}{{1}}
\noindent\textbf{{Supplement}}

\vspace{{6pt}}

\noindent\textbf{{eTable 1.}} Survey Items and Analytic Coding, 2023 National Youth Risk Behavior Survey

\vspace{{6pt}}

\noindent\textit{{Response rate.}} The 2023 National YRBS had an overall response rate of 35.4\%
(school response rate 49.8\% $\times$ student response rate 71.0\%).$^{{4}}$

\vspace{{12pt}}

{supplement_tex}

\end{{document}}
"""

with open(os.path.join(BASE, 'draft-v2.tex'), 'w') as f:
    f.write(tex)
print("  Wrote draft-v2.tex")


# ── Compile LaTeX ─────────────────────────────────────────────
print("Compiling LaTeX...")
for _ in range(2):
    subprocess.run(['pdflatex', '-interaction=nonstopmode', 'draft-v2.tex'],
                   cwd=BASE, capture_output=True)
pdf = os.path.join(BASE, 'draft-v2.pdf')
print(f"  PDF: {os.path.getsize(pdf):,} bytes" if os.path.exists(pdf) else "  PDF FAILED")


print("\nDone.")
