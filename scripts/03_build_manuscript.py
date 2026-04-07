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
strat_rr  = pd.read_csv(os.path.join(DATA, "stratified_rr.csv"))
coll_rr   = pd.read_csv(os.path.join(DATA, "collapsed_rr.csv"))
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
    ("Current cigarette use", "Substance Use", "j",
     "Smoked a cigarette $\\geq$1 day (past 30 days)."),
    ("Current e-cigarette use", "Substance Use", "k",
     "Used an electronic vapor product $\\geq$1 day (past 30 days)."),
    ("Current alcohol use", "Substance Use", "l",
     "Had $\\geq$1 drink of alcohol $\\geq$1 day (past 30 days)."),
    ("Current marijuana use", "Substance Use", "m",
     "Used marijuana $\\geq$1 time (past 30 days)."),
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
    "Current cigarette use": "QN33",
    "Current e-cigarette use": "QN36",
    "Current alcohol use": "QN42",
    "Current marijuana use": "QN48",
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

# ── Load non-White prevalences and text stats from R ──────────
print("Loading non-White prevalences and text stats...")
nw_prev_df = pd.read_csv(os.path.join(DATA, "nw_prevalences.csv"))
sui_stats  = pd.read_csv(os.path.join(DATA, "suicide_text_stats.csv"))
sui_never  = sui_stats[sui_stats['group'] == 'Never'].iloc[0]['pct']
sui_rs     = sui_stats[sui_stats['group'] == 'Rarely/Sometimes'].iloc[0]['pct']
sui_ma     = sui_stats[sui_stats['group'] == 'Most/Always'].iloc[0]['pct']

# Group sizes from raw data (for eTable panels)
raw_full = pd.read_csv(os.path.join(DATA, "yrbs2023_parsed.csv"))
total_n = len(raw_full)
raw = raw_full[raw_full['Q23'].notna() & raw_full['weight'].notna() & (raw_full['weight'] > 0)]
nw = raw[(raw['raceeth'].notna()) & (raw['raceeth'] != 5)]
nw_n = len(nw)
group_ns = {'All': len(raw), 'Non-White': nw_n}
for _rn, _rv in [('White', [5]), ('Black', [3]), ('Hispanic', [6]), ('Asian', [2]),
                  ('AI/AN', [1]), ('NH/PI', [4]), ('Multiple', [7, 8])]:
    group_ns[_rn] = len(raw[raw['raceeth'].isin(_rv)])
nw_total_w = nw['weight'].astype(float).sum()
nw_ma_w = nw[nw['Q23'].isin([4, 5])]['weight'].astype(float).sum()
nw_ma_p = 100 * nw_ma_w / nw_total_w


# ── Pull key numbers for text ─────────────────────────────────
ov = racism_p[racism_p['group'] == 'Overall'].iloc[0]
n_valid = int(ov['n'])
any_p, any_lo, any_hi = ov['any_pct'], ov['any_lo'], ov['any_hi']

race_pcts = {}
for _, row in racism_p.iterrows():
    race_pcts[row['group']] = row['any_pct']

# Bullied at school collapsed RRs from proper 3-level regression
def _get_collapsed(grp, outcome, level):
    """Get RR, lo, hi from collapsed_rr.csv."""
    row = coll_rr[(coll_rr['group'] == grp) & (coll_rr['outcome'] == outcome) &
                  (coll_rr['racism_level'] == level)]
    if len(row) and not pd.isna(row.iloc[0]['rr']):
        r = row.iloc[0]
        return r['rr'], r['rr_lo'], r['rr_hi']
    return np.nan, np.nan, np.nan

bul_rs_rr, bul_rs_lo, bul_rs_hi = _get_collapsed("All", "Bullied at school", "Rarely/Sometimes")
bul_ma_rr, bul_ma_lo, bul_ma_hi = _get_collapsed("All", "Bullied at school", "Most/Always")

# Race-stratified text (Asian, Black) — collapsed groups from proper 3-level regression
race_parts = []
race_parts_plain = []
for race in ['Asian', 'Black']:
    rs_rr, rs_lo, rs_hi = _get_collapsed(race, "Bullied at school", "Rarely/Sometimes")
    ma_rr, ma_lo, ma_hi = _get_collapsed(race, "Bullied at school", "Most/Always")
    # LaTeX version
    race_parts.append(
        f'{race} children (rarely or sometimes RR {rs_rr:.2f}, '
        f'95\\% CI {rs_lo:.2f}--{rs_hi:.2f}; '
        f'most of the time or always RR {ma_rr:.2f}, '
        f'95\\% CI {ma_lo:.2f}--{ma_hi:.2f})')
    # Plain-text version for word count
    race_parts_plain.append(
        f'{race} children (rarely or sometimes RR {rs_rr:.2f}, '
        f'95% CI {rs_lo:.2f}\u2013{rs_hi:.2f}; '
        f'most of the time or always RR {ma_rr:.2f}, '
        f'95% CI {ma_lo:.2f}\u2013{ma_hi:.2f})')
race_text = ' and '.join(race_parts)
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

# Compute Ns per racism level for table headers
racism_level_ns = {}
nw_racism_level_ns = {}
for lev_val, lev_name in [(1, 'Never'), (2, 'Rarely'), (3, 'Sometimes'), (4, 'Most'), (5, 'Always')]:
    racism_level_ns[lev_name] = int(raw[raw['Q23'] == lev_val].shape[0])
    nw_racism_level_ns[lev_name] = int(nw[nw['Q23'] == lev_val].shape[0])

panel_a = make_panel_tabular(prev_df)
panel_b = make_panel_tabular(nw_prev_df)

table_tex = rf"""\begin{{landscape}}
\begin{{table}}[p]
\centering
\caption{{Weighted Prevalence (\%) of Health Outcomes by Frequency of School-Based Racial Discrimination, 2023 YRBS}}
\label{{tab:main}}
\footnotesize
\renewcommand{{\arraystretch}}{{0.88}}

\textbf{{Panel A: Full Sample ($N = {n_valid:,}$)}}
\vspace{{4pt}}

\begin{{tabular}}{{l r ccccc}}
\toprule
 & & \multicolumn{{5}}{{c}}{{Weighted Prevalence, \% (95\% CI)}} \\
\cmidrule(lr){{3-7}}
Outcome & $N$ & Never & Rarely & Sometimes & Most of the time & Always \\
 & & ($n = {racism_level_ns['Never']:,}$) & ($n = {racism_level_ns['Rarely']:,}$) & ($n = {racism_level_ns['Sometimes']:,}$) & ($n = {racism_level_ns['Most']:,}$) & ($n = {racism_level_ns['Always']:,}$) \\
\midrule
{panel_a}
\bottomrule
\end{{tabular}}

\vspace{{8pt}}

\textbf{{Panel B: Non-White Students ($N = {nw_n:,}$)}}
\vspace{{2pt}}

\begin{{tabular}}{{l r ccccc}}
\toprule
 & & \multicolumn{{5}}{{c}}{{Weighted Prevalence, \% (95\% CI)}} \\
\cmidrule(lr){{3-7}}
Outcome & $N$ & Never & Rarely & Sometimes & Most of the time & Always \\
 & & ($n = {nw_racism_level_ns['Never']:,}$) & ($n = {nw_racism_level_ns['Rarely']:,}$) & ($n = {nw_racism_level_ns['Sometimes']:,}$) & ($n = {nw_racism_level_ns['Most']:,}$) & ($n = {nw_racism_level_ns['Always']:,}$) \\
\midrule
{panel_b}
\bottomrule
\end{{tabular}}

\vspace{{4pt}}
\begin{{minipage}}{{\linewidth}}
\scriptsize
\textit{{Note.}} CI = confidence interval. $N$ = unweighted analytic sample.
$n$ = unweighted students at each level of school-based racial discrimination.
Prevalences are weighted to the YRBS complex three-stage cluster design.
\end{{minipage}}

\end{{table}}
\end{{landscape}}"""


# ── Figure helper ──────────────────────────────────────────────
DOMAIN_COLORS_FIG = {
    'Mental Health': '#2166ac',
    'Violence': '#b2182b',
    'Substance Use': '#e66101',
}

def make_dose_response_figure(rr_data, save_path):
    """Generate faceted dose-response figure from risk ratio data."""
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (label, domain, _, _) in enumerate(OUTCOMES):
        ax = axes[idx]
        x = list(range(5))
        y, lo, hi = [], [], []

        y.append(1.0); lo.append(1.0); hi.append(1.0)

        for lev in ["Rarely", "Sometimes", "Most", "Always"]:
            row = rr_data[(rr_data['outcome'] == label) & (rr_data['racism_level'] == lev)]
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
        ax.tick_params(axis='y', labelsize=10)
        ax.set_facecolor('white')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
            spine.set_color('#333333')
        ax.grid(True, which='major', axis='both', color='#cccccc', linewidth=0.5, linestyle='-')
        ax.set_axisbelow(True)

        short = label
        if len(short) > 22:
            words = short.split(); mid = len(words) // 2
            short = ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
        ax.set_title(short, fontsize=12, fontweight='bold', pad=8)
        if idx % 4 == 0:
            ax.set_ylabel('Adjusted\nRisk Ratio', fontsize=11)

    fig.supxlabel('Frequency of School-Based Racial Discrimination', fontsize=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ── Generate figures ──────────────────────────────────────────
print("Generating figures...")
make_dose_response_figure(rr_df, os.path.join(FIGS, 'figure1.pdf'))
print("  Saved figure1")

nw_rr = strat_rr[strat_rr['group'] == 'Non-White']
make_dose_response_figure(nw_rr, os.path.join(FIGS, 'efigure1.pdf'))
print("  Saved efigure1")


# ── Build supplement eTable (as plain string to avoid rf-string escaping) ──
supplement_tex = r"""
\begin{landscape}
\singlespacing
\noindent\textbf{Supplement}\par
\vspace{6pt}
\noindent The 2023 National YRBS had an overall response rate of 35.4\% (school response rate 49.8\% $\times$ student response rate 71.0\%).\cite{cdc2023yrbs}\par
\vspace{8pt}
\noindent\textbf{eTable 1.} Survey Items and Analytic Coding, 2023 National Youth Risk Behavior Survey\par
\vspace{6pt}
\footnotesize
\renewcommand{\arraystretch}{1.1}
\begin{longtable}{>{\raggedright\arraybackslash}p{2.2in} >{\raggedright\arraybackslash}p{4.0in} >{\raggedright\arraybackslash}p{2.8in}}
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

\textit{Exposure} \\
\addlinespace[2pt]

School-based racism (Q23) &
``During your life, how often have you felt that you were treated badly or unfairly
in school because of your race or ethnicity?'' &
1 = Never; 2 = Rarely; 3 = Sometimes; 4 = Most of the time; 5 = Always.
Coded as categorical (5 levels; reference = Never). \\

\addlinespace[6pt]
\textit{Outcomes --- Mental Health} \\
\addlinespace[2pt]

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

\addlinespace[6pt]
\textit{Outcomes --- Violence} \\
\addlinespace[2pt]

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

\addlinespace[6pt]
\textit{Outcomes --- Substance Use} \\
\addlinespace[2pt]

Current cigarette use (QN33) &
``During the past 30 days, on how many days did you smoke cigarettes?'' &
0 / 1--2 / 3--5 / 6--9 / 10--19 / 20--29 / All 30 days.
Coded as binary (1 if $\geq$1 day). \\
\addlinespace[3pt]

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
\addlinespace[3pt]

Current marijuana use (QN48) &
``During the past 30 days, how many times did you use marijuana?'' &
0 / 1--2 / 3--9 / 10--19 / 20--39 / 40+ times.
Coded as binary (1 if $\geq$1 time). \\

\addlinespace[6pt]
\textit{Covariates} \\
\addlinespace[2pt]

Sex (Q2) &
``What is your sex?'' &
Female / Male. Coded as binary (1 = Female). \\
\addlinespace[3pt]

Age (Q1) &
``How old are you?'' &
12 years or younger / 13 / 14 / 15 / 16 / 17 / 18 or older.
Coded as categorical (5 levels: $\leq$14, 15, 16, 17, 18+; ages $\leq$14 collapsed
because of sparse counts at 12 and 13). \\
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
\end{landscape}
"""

# ── Generate eTable 2 (race-stratified ARRs) ─────────────────
print("Generating eTable 2...")

ETABLE_GROUPS = [
    ('All', 'All Students', rr_df),
    ('Non-White', 'Non-White Students', strat_rr[strat_rr['group'] == 'Non-White']),
    ('White', 'White Students', strat_rr[strat_rr['group'] == 'White']),
    ('Black', 'Black Students', strat_rr[strat_rr['group'] == 'Black']),
    ('Hispanic', 'Hispanic Students', strat_rr[strat_rr['group'] == 'Hispanic']),
    ('Asian', 'Asian Students', strat_rr[strat_rr['group'] == 'Asian']),
    ('AI/AN', 'American Indian/Alaska Native Students', strat_rr[strat_rr['group'] == 'AI/AN']),
    ('NH/PI', 'Native Hawaiian/Pacific Islander Students', strat_rr[strat_rr['group'] == 'NH/PI']),
    ('Multiple', 'Multiracial Students', strat_rr[strat_rr['group'] == 'Multiple']),
]

def make_rr_panel_rows(rr_data):
    """Build tabular body rows for one eTable panel (ARRs)."""
    rows = []
    current_domain = None
    for label, domain, _, _ in OUTCOMES:
        if domain != current_domain:
            if current_domain is not None:
                rows.append(r'\addlinespace[4pt]')
            rows.append(f'\\textit{{{domain}}} \\\\')
            current_domain = domain

        sub = rr_data[rr_data['outcome'] == label]
        n_val = int(sub['n'].iloc[0]) if len(sub) else 0

        cells = ['1 [Ref]']
        for lev in ["Rarely", "Sometimes", "Most", "Always"]:
            row = sub[sub['racism_level'] == lev]
            if len(row) and not pd.isna(row.iloc[0]['rr']):
                r = row.iloc[0]
                # Suppress extreme estimates from tiny samples
                if r['rr'] > 50 or r['rr_hi'] > 100:
                    cells.append('---')
                else:
                    cells.append(f'{r["rr"]:.2f} ({r["rr_lo"]:.2f}--{r["rr_hi"]:.2f})')
            else:
                cells.append('---')

        rows.append(f'\\quad {label} & {n_val:,} & '
                     + ' & '.join(cells) + r' \\')
    return '\n'.join(rows)

def make_etable2_panel_block(grp_label, grp_n, grp_data):
    """Build one stand-alone tabular block for a single race-stratified panel.

    Each panel is a tabular preceded by an inline bold header. The
    header is glued to the tabular via a non-breakable parindent + nobreak
    construct so they always stay together. Tabular itself is atomic and
    cannot break across pages.
    """
    panel_rows = make_rr_panel_rows(grp_data)
    return (
        r'\nobreak\noindent\textbf{' f'{grp_label} ($N = {grp_n:,}$)' r'}' '\n'
        r'\nopagebreak\par\nopagebreak\vspace{1pt}\nopagebreak' '\n'
        r'\nobreak\noindent\begin{tabular}{l r ccccc}' '\n'
        r'\toprule' '\n'
        r' & & \multicolumn{5}{c}{Adjusted Risk Ratio (95\% CI)} \\' '\n'
        r'\cmidrule(lr){3-7}' '\n'
        r'Outcome & $N$ & Never & Rarely & Sometimes & Most of the time & Always \\' '\n'
        r'\midrule' '\n'
        f'{panel_rows}' '\n'
        r'\bottomrule' '\n'
        r'\end{tabular}\par\vspace{8pt}' '\n'
    )

etable2_panels = []
for grp_key, grp_label, grp_data in ETABLE_GROUPS:
    grp_n = group_ns.get(grp_key, 0)
    etable2_panels.append(make_etable2_panel_block(grp_label, grp_n, grp_data))

etable2_body = '\n'.join(etable2_panels)

etable2_tex = (
    r'\begin{landscape}' '\n'
    r'\footnotesize' '\n'
    r'\noindent\textbf{eTable 2.} Adjusted Risk Ratios for Health Outcomes by Frequency of School-Based Racial '
    r'Discrimination, Stratified by Race/Ethnicity, 2023 YRBS\par' '\n'
    r'\vspace{6pt}' '\n'
    f'{etable2_body}\n'
    r'\normalsize' '\n'
    r'\end{landscape}'
)

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
    f"Youth Risk Behavior Surveillance System (YRBS; N = {total_n:,}; overall response rate "
    f"35.4%), a biannual survey administered in US high schools on behavioral health "
    f"topics. The survey uses a three-stage cluster design to capture a representative "
    f"sample of high school students. The survey includes measures of mental health, substance "
    f"use, and violence victimization, among other important health behaviors (eTable 1 in the "
    f"Supplement). In 2023, the national YRBS introduced a measure of children\u2019s experience "
    f"with racism at school. Children were asked \u201cHow often have you felt that you were treated "
    f"badly or unfairly in school because of your race or ethnicity?\u201d with possible responses "
    f"\u201cNever,\u201d \u201cRarely,\u201d \u201cSometimes,\u201d \u201cMost of the time,\u201d and \u201cAlways.\u201d "
    f"First, the unadjusted prevalence of several risky behaviors is reported by level of "
    f"experience with racism. Then, the relative risk of each of these behaviors was "
    f"estimated via survey-weighted logistic regression, accounting for sex, age, race/ethnicity, "
    f"English language proficiency, and grades, relative to those who experienced no racism. "
    f"These analyses were conducted collectively and by student race/ethnicity. Estimates are "
    f"presented as risk ratios, which are computed using random draws from the regression\u2019s "
    f"variance-covariance matrix and holding confounders at their means. "
    f"All analyses account for the YRBS complex survey design to produce nationally "
    f"representative estimates and correct standard errors. "
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
    f"{race_pcts.get('White', 0):.1f}% of White children (Table 1). "
    f"{nw_ma_p:.1f}% of non-White children reported experiencing racism at school most of the "
    f"time or always. "
    f"Most negative health behaviors exhibit a dose-response relationship with children\u2019s "
    f"experience with racism. For example, suicidal "
    f"ideation has a prevalence of {sui_never:.1f}% among children who report never "
    f"experiencing racism, {sui_rs:.1f}% among children who report experiencing racism rarely "
    f"or sometimes, and {sui_ma:.1f}% among children who report experiencing racism most of "
    f"the time or always. "
    f"These patterns persist after accounting for potential confounders (Figure 1). After controlling for "
    f"sex, age, race/ethnicity, English language proficiency, and grades, those who experience "
    f"racism rarely or sometimes (RR {bul_rs_rr:.2f}, 95% CI "
    f"{bul_rs_lo:.2f}\u2013{bul_rs_hi:.2f}) and those who experience racism most of the time "
    f"or always (RR {bul_ma_rr:.2f}, 95% CI "
    f"{bul_ma_lo:.2f}\u2013{bul_ma_hi:.2f}) are significantly more likely to be bullied at "
    f"school compared with those who never experience racism. The response is strongest for "
    f"{race_text_plain} (eTable 2 in the Supplement).",
    # Discussion
    "A significant proportion of US high school children report experiencing racism in schools, "
    "and these experiences correlate significantly and in a dose-response relationship with "
    "several negative health outcomes. "
    "Racism is a well-recognized threat to public health. These results extend that "
    "finding to children in schools and merit a well-coordinated response from public "
    "health professionals. Recent studies find that a composite measure of racial mistreatment "
    "and school connection is strongly correlated with substance use and mental health behaviors "
    "among American adolescents. The present study complements these findings by examining a "
    "broader set of health outcomes in a larger, nationally representative sample. "
    "Policy-makers, educators, and clinicians may design "
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
\usepackage{{needspace}}
\usepackage[numbers,super,comma]{{natbib}}
\usepackage[hidelinks]{{hyperref}}
\captionsetup{{font=footnotesize, labelfont=bf}}
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

\noindent\textbf{{Corresponding Author:}} Theodore L. Caputi, MIT Department of Economics,
50 Memorial Dr, Cambridge, MA 02142 (tcaputi@gmail.com, https://www.tlcaputi.com).

\vspace{{12pt}}

\noindent\textbf{{Conflict of Interest Disclosures:}} TLC declares an equity
interest in Data Science Solutions, a public health consulting firm not
involved with this paper.

\vspace{{12pt}}

\noindent\textbf{{Word Count:}} {word_count}

\clearpage

%% ── BODY ───────────────────────────────────────────────────
\noindent\textbf{{Introduction}}

Racism is one of the best documented and most powerful social determinants of
health.\cite{{williams2013racism,paradies2015racism}} However, there has been relatively little research studying the role
that racism plays in the health of children.\cite{{priest2013racism}} In this study, we examine the
relationship between experiences with racism at school and important health behaviors
among a nationally representative sample of US high school students.

\vspace{{6pt}}

\noindent\textbf{{Methods}}

This study uses data from the Centers for Disease Control and Prevention's 2023
Youth Risk Behavior Surveillance System (YRBS; $N = {total_n:,}$; overall response rate
35.4\%), a biannual survey administered in US high schools on behavioral health
topics.\cite{{cdc2023yrbs}} The survey uses a three-stage cluster design to capture a representative
sample of high school students. The survey includes measures of mental health, substance
use, and violence victimization, among other important health behaviors (eTable 1 in the
Supplement). In 2023, the national YRBS introduced a measure of children's experience
with racism at school. Children were asked ``How often have you felt that you were treated
badly or unfairly in school because of your race or ethnicity?'' with possible responses
``Never,'' ``Rarely,'' ``Sometimes,'' ``Most of the time,'' and ``Always.''

First, the unadjusted prevalence of several risky behaviors is reported by level of
experience with racism. Then, the relative risk of each of these
behaviors was estimated via survey-weighted logistic regression, accounting for sex, age,
race/ethnicity, English language proficiency, and grades, relative to those who
experienced no racism.
These analyses were conducted collectively and by student race/ethnicity. Estimates are
presented as risk ratios, which are computed using random draws from the regression's
variance-covariance matrix and holding confounders at their means.\cite{{king2000making}}
All analyses account for the YRBS complex survey design to produce nationally
representative estimates and correct standard errors.

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
{race_pcts.get('White', 0):.1f}\% of White children (\autoref{{tab:main}}).
{nw_ma_p:.1f}\% of non-White children reported experiencing racism at school most of the
time or always.

Most negative health behaviors exhibit a dose-response relationship with children's
experience with racism. For example, suicidal
ideation has a prevalence of {sui_never:.1f}\% among children who report never
experiencing racism, {sui_rs:.1f}\% among children who report experiencing racism rarely
or sometimes, and {sui_ma:.1f}\% among children who report experiencing racism most of
the time or always.

These patterns persist after accounting for potential confounders (\autoref{{fig:rr}}). After controlling for
sex, age, race/ethnicity, English language proficiency, and grades, those who experience
racism rarely or sometimes (RR {bul_rs_rr:.2f}, 95\% CI
{bul_rs_lo:.2f}--{bul_rs_hi:.2f}) and those who experience racism most of the time
or always (RR {bul_ma_rr:.2f}, 95\% CI
{bul_ma_lo:.2f}--{bul_ma_hi:.2f}) are significantly more likely to be bullied at
school compared with those who never experience racism. The response is strongest for
{race_text} (eTable 2 in the Supplement).

\vspace{{6pt}}

\noindent\textbf{{Discussion}}

A significant proportion of US high school children report experiencing racism in schools,
and these experiences correlate significantly and in a dose-response relationship with
several negative health outcomes.

Racism is a well-recognized threat to public health.\cite{{williams2013racism,paradies2015racism}} These results extend that
finding to children in schools\cite{{priest2013racism}} and merit a well-coordinated response from public
health professionals. Recent studies find that a composite measure of racial mistreatment and school connection is strongly correlated with substance use and mental health behaviors among American adolescents. \cite{{azagba2025school,azagba2026belonging}} The present study complements these findings by examining a broader
set of health outcomes in a larger, nationally representative sample. Policy-makers, educators, and clinicians
may design interventions to reduce racism among young people and to screen children who
experience racism at school for negative health behaviors.\cite{{trent2019impact}}

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
\bibliographystyle{{unsrtnat}}
\bibliography{{references}}

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

{supplement_tex}

{etable2_tex}

\begin{{landscape}}
\begin{{figure}}[H]
  \centering
  \includegraphics[width=\linewidth]{{figures/efigure1.pdf}}
  \caption*{{\textbf{{eFigure 1.}} Adjusted risk ratios for health outcomes by frequency of school-based
  racism among non-White US high school students, 2023 YRBS ($N = {nw_n:,}$).
  Risk ratios from logistic regression computed via simulation from the
  variance-covariance matrix (King et al, 2000), adjusting for sex, age,
  race/ethnicity, English language proficiency, and grades.
  Shaded bands indicate 95\% confidence intervals. Reference category is Never.}}
  \label{{fig:efig1}}
\end{{figure}}

\vspace{{6pt}}

\noindent\textbf{{Data and Code Availability.}} Data are publicly available from the Centers
for Disease Control and Prevention.\cite{{cdc2023yrbs}} Code to download the data and reproduce all analyses is
available at \url{{https://github.com/tlcaputi/yrbs-racism}}.
\end{{landscape}}

\end{{document}}
"""

with open(os.path.join(BASE, 'draft-v2.tex'), 'w') as f:
    f.write(tex)
print("  Wrote draft-v2.tex")


# ── Compile LaTeX (pdflatex → bibtex → pdflatex × 2) ─────────
print("Compiling LaTeX...")
subprocess.run(['pdflatex', '-interaction=nonstopmode', 'draft-v2.tex'],
               cwd=BASE, capture_output=True)
subprocess.run(['bibtex', 'draft-v2'], cwd=BASE, capture_output=True)
for _ in range(2):
    subprocess.run(['pdflatex', '-interaction=nonstopmode', 'draft-v2.tex'],
                   cwd=BASE, capture_output=True)
pdf = os.path.join(BASE, 'draft-v2.pdf')
print(f"  PDF: {os.path.getsize(pdf):,} bytes" if os.path.exists(pdf) else "  PDF FAILED")


print("\nDone.")
