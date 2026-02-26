# Racism in Schools and Health Among US Adolescents

Evidence from the 2023 Youth Risk Behavior Survey.

## Requirements

- Python 3.8+ with `pandas`, `numpy`, `matplotlib`
- R 4.0+ with [`zelig2`](https://github.com/tlcaputi/zelig2) and `survey`
- LaTeX distribution (for PDF compilation)
- `curl` (for data download)

### R package installation

```r
install.packages("survey")
devtools::install_github("tlcaputi/zelig2")
```

## Run

```bash
bash run.sh
```

### Pipeline

| Step | Script | Description |
|------|--------|-------------|
| 0 | `scripts/00_download_data.sh` | Download YRBS 2023 data from CDC |
| 1 | `scripts/01_parse_yrbs.py` | Parse fixed-width ASCII into CSV |
| 2 | `scripts/02_analysis.R` | Logistic regression + King simulation |
| 3 | `scripts/03_build_manuscript.py` | Generate table, figure, LaTeX, PDF |

### Outputs

- `draft-v2.pdf` — Compiled LaTeX manuscript
- `figures/figure1.pdf` — Dose-response figure
- `data/prevalences.csv` — Weighted prevalences by outcome and racism level
- `data/risk_ratios.csv` — Adjusted risk ratios
- `data/race_stratified_rr.csv` — Race-stratified risk ratios
- `data/racism_prevalence.csv` — Racism prevalence by race

## Author

Theodore L. Caputi
Massachusetts Institute of Technology, Department of Economics
tcaputi@gmail.com | [tlcaputi.com](https://www.tlcaputi.com)
