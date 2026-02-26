#!/usr/bin/env python3
"""
Parse YRBS 2023 national fixed-width ASCII data file into a pandas DataFrame.
Column positions taken directly from 2023XXH-SPSS.sps syntax file.
"""

import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
DAT_FILE = os.path.join(DATA_DIR, "XXH2023_YRBS_Data.dat")

# Fixed-width column specifications from SPSS syntax
# Format: (colname, start_col_0indexed, end_col_exclusive)
# SPSS uses 1-based cols; Python slices are 0-based.

# Key variables needed:
# Q1  17-17   Age
# Q2  18-18   Sex
# Q3  19-19   Grade
# Q4  20-20   Hispanic/Latino
# Q5  21-28   Race (8 chars)
# Q17 48-48   Physical fight at school
# Q22 53-53   Physical dating violence  (note: Q23 is 54-54)
# Q23 54-54   Treated unfairly in school b/c race/ethnicity  *** RACISM EXPOSURE ***
# Q24 55-55   Bullied at school
# Q25 56-56   Electronic bullying
# Q26 57-57   Sad or hopeless
# Q27 58-58   Considered suicide
# Q28 59-59   Made suicide plan
# Q29 60-60   Attempted suicide
# Q30 61-61   Injurious suicide attempt
# Q36 67-67   Current electronic vapor product use
# Q42 73-73   Current alcohol use
# Q43 74-74   Current binge drinking
# Q48 79-79   Current marijuana use
# Q64 95-95   Sexual identity
# Q65 96-96   Transgender
# Q84 115-115 Current mental health (most of time/always not good)
# Q85 116-116 Hours of sleep
# Q12 43-43   Weapon carrying at school
# QN23 200-200 Dichotomous: felt treated unfairly
# QN24 201-201 Bullied on school property
# QN25 202-202 Electronically bullied
# QN26 203-203 Sad or hopeless (dichotomous)
# QN27 204-204 Seriously considered suicide
# QN28 205-205 Made suicide plan
# QN29 206-206 Attempted suicide
# QN36 213-213 Currently used EVP
# QN42 219-219 Currently drank alcohol
# QN43 220-220 Currently binge drinking
# QN46 223-223 Ever used marijuana
# QN48 225-225 Currently used marijuana
# QN84 261-261 Mental health not good most/always
# QN12 189-189 Carried weapon on school property
# QN17 194-194 Physical fight on school property
# weight 388-397
# stratum 398-400
# psu 401-406
# raceeth 414-415

colspecs = [
    ('site',     0,  3),    # site 1-3
    ('Q1',      16, 17),    # Q1 17-17
    ('Q2',      17, 18),    # Q2 18-18
    ('Q3',      18, 19),    # Q3 19-19
    ('Q4',      19, 20),    # Q4 20-20
    ('Q5',      20, 28),    # Q5 21-28 (8 chars)
    ('Q12',     42, 43),    # Q12 43-43
    ('Q16',     46, 47),    # Q16 47-47
    ('Q17',     47, 48),    # Q17 48-48
    ('Q23',     53, 54),    # Q23 54-54  RACISM
    ('Q24',     54, 55),    # Q24 55-55
    ('Q25',     55, 56),    # Q25 56-56
    ('Q26',     56, 57),    # Q26 57-57  sad/hopeless
    ('Q27',     57, 58),    # Q27 58-58  considered suicide
    ('Q28',     58, 59),    # Q28 59-59  suicide plan
    ('Q29',     59, 60),    # Q29 60-60  attempted suicide
    ('Q30',     60, 61),    # Q30 61-61  injurious attempt
    ('Q36',     66, 67),    # Q36 67-67  current EVP
    ('Q42',     72, 73),    # Q42 73-73  current alcohol
    ('Q43',     73, 74),    # Q43 74-74  current binge drinking
    ('Q46',     76, 77),    # Q46 77-77  ever marijuana
    ('Q48',     78, 79),    # Q48 79-79  current marijuana
    ('Q64',     94, 95),    # Q64 95-95  sexual identity
    ('Q65',     95, 96),    # Q65 96-96  transgender
    ('Q84',    114,115),    # Q84 115-115 current mental health
    ('Q85',    115,116),    # Q85 116-116 sleep hours
    # Dichotomous QN variables (1=yes, " "=missing)
    ('QN12',   188,189),    # QN12 189-189 weapon at school
    ('QN17',   193,194),    # QN17 194-194 physical fight at school
    ('QN23',   199,200),    # QN23 200-200 treated unfairly (dichotomous)
    ('QN24',   200,201),    # QN24 201-201 bullied at school
    ('QN25',   201,202),    # QN25 202-202 electronically bullied
    ('QN26',   202,203),    # QN26 203-203 sad or hopeless
    ('QN27',   203,204),    # QN27 204-204 considered suicide
    ('QN28',   204,205),    # QN28 205-205 suicide plan
    ('QN29',   205,206),    # QN29 206-206 attempted suicide
    ('QN36',   212,213),    # QN36 213-213 current EVP
    ('QN42',   218,219),    # QN42 219-219 current alcohol
    ('QN43',   219,220),    # QN43 220-220 binge drinking
    ('QN46',   222,223),    # QN46 223-223 ever marijuana
    ('QN48',   224,225),    # QN48 225-225 current marijuana
    ('QN84',   260,261),    # QN84 261-261 mental health not good
    ('QN87',   263,264),    # QN87 264-264 grades mostly A's or B's
    ('QN107',  283,284),    # QN107 284-284 speak English well or very well
    # Survey design variables
    ('weight',  387,397),   # weight 388-397
    ('stratum', 397,400),   # stratum 398-400
    ('psu',     400,406),   # psu 401-406
    ('raceeth', 413,415),   # raceeth 414-415
]

print(f"Parsing {DAT_FILE}")
print(f"File size: {os.path.getsize(DAT_FILE):,} bytes")

# Read the fixed-width file
rows = []
with open(DAT_FILE, 'r') as f:
    for lineno, line in enumerate(f):
        row = {}
        for (name, start, end) in colspecs:
            val = line[start:end] if len(line) > start else ' ' * (end - start)
            row[name] = val
        rows.append(row)

df = pd.DataFrame(rows)
print(f"Rows read: {len(df):,}")
print(f"Columns: {list(df.columns)}")

# Convert numeric variables
def to_numeric(s):
    s = s.strip()
    if s == '' or s == ' ':
        return np.nan
    try:
        return float(s)
    except:
        return np.nan

numeric_cols = [c for c in df.columns if c not in ('site', 'Q5')]
for col in numeric_cols:
    df[col] = df[col].apply(to_numeric)

# Weight must be positive for survey analysis
df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
df = df[df['weight'].notna() & (df['weight'] > 0)].copy()
print(f"Rows with valid weight: {len(df):,}")

# Check racism variable distribution
print("\nQ23 (racism) distribution (raw):")
print(df['Q23'].value_counts(dropna=False).sort_index())

print("\nraceeth distribution:")
print(df['raceeth'].value_counts(dropna=False).sort_index())

# Save parsed data
out_file = os.path.join(DATA_DIR, "yrbs2023_parsed.csv")
df.to_csv(out_file, index=False)
print(f"\nSaved parsed data to: {out_file}")
print(f"Shape: {df.shape}")
