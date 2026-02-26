#!/usr/bin/env Rscript
# ──────────────────────────────────────────────────────────────
# YRBS 2023 Racism & Health — Analysis
# Logistic regression + King et al. (2000) simulation via zelig2
# Outputs CSV files consumed by 03_build_manuscript.py
# ──────────────────────────────────────────────────────────────
suppressPackageStartupMessages({
  library(zelig2)
  library(survey)
})

set.seed(20230101)  # reproducibility

BASE <- tryCatch(
  normalizePath(file.path(dirname(sys.frame(1)$ofile), "..")),
  error = function(e) normalizePath(".")
)
DATA_DIR <- file.path(BASE, "data")

# ── 1. Load & recode ─────────────────────────────────────────
cat("Loading data...\n")
df <- read.csv(file.path(DATA_DIR, "yrbs2023_parsed.csv"))
cat(sprintf("  %s rows\n", format(nrow(df), big.mark = ",")))

# Racism factor (ref = Never)
df$racism_f <- factor(df$Q23, levels = 1:5,
                      labels = c("Never", "Rarely", "Sometimes", "Most", "Always"))

# Covariates (matching Caputi et al. JAMA 2017)
df$female       <- ifelse(df$Q2 == 1, 1, ifelse(df$Q2 == 2, 0, NA))
df$age_cat      <- ifelse(df$Q1 >= 1 & df$Q1 <= 7, df$Q1, NA)
df$race_f       <- factor(ifelse(df$raceeth >= 1 & df$raceeth <= 8, df$raceeth, NA))
df$english_prof <- ifelse(df$QN107 == 1, 1, ifelse(df$QN107 == 2, 0, NA))
df$good_grades  <- ifelse(df$QN87 == 1, 1, ifelse(df$QN87 == 2, 0, NA))

# Race category for stratification
race_map <- c("1" = "AI/AN", "2" = "Asian", "3" = "Black", "4" = "NH/PI",
              "5" = "White", "6" = "Hispanic", "7" = "Multiple", "8" = "Multiple")
df$race_cat <- race_map[as.character(df$raceeth)]

# Recode QN outcomes: 1=yes, 2=no → 1/0
outcome_defs <- list(
  list(qn = "QN26", label = "Persistent sadness or hopelessness"),
  list(qn = "QN27", label = "Seriously considered suicide"),
  list(qn = "QN28", label = "Made a suicide plan"),
  list(qn = "QN29", label = "Attempted suicide"),
  list(qn = "QN24", label = "Bullied at school"),
  list(qn = "QN25", label = "Electronically bullied"),
  list(qn = "QN17", label = "Physical fight at school"),
  list(qn = "QN12", label = "Carried weapon at school"),
  list(qn = "QN36", label = "Current e-cigarette use"),
  list(qn = "QN42", label = "Current alcohol use")
)

for (od in outcome_defs) {
  col <- paste0(od$qn, "_bin")
  df[[col]] <- ifelse(df[[od$qn]] == 1, 1, ifelse(df[[od$qn]] == 2, 0, NA))
}

# ── 2. Weighted prevalence ───────────────────────────────────
cat("Computing weighted prevalences...\n")

wprev <- function(y, w) {
  ok <- !is.na(y) & !is.na(w) & w > 0
  if (sum(ok) < 5) return(c(pct = NA, lo = NA, hi = NA, n = 0))
  yy <- y[ok]; ww <- w[ok]
  p <- sum(yy * ww) / sum(ww)
  se <- sqrt(sum(((yy - p) * ww)^2) / sum(ww)^2)
  c(pct = 100 * p,
    lo  = 100 * max(0, p - 1.96 * se),
    hi  = 100 * min(1, p + 1.96 * se),
    n   = sum(ok))
}

prev_rows <- list()
for (od in outcome_defs) {
  col <- paste0(od$qn, "_bin")
  for (lev in c("Never", "Rarely", "Sometimes", "Most", "Always")) {
    sub <- df[!is.na(df$racism_f) & df$racism_f == lev, ]
    wp <- wprev(sub[[col]], sub$weight)
    prev_rows[[length(prev_rows) + 1]] <- data.frame(
      outcome = od$label, outcome_var = col, racism_level = lev,
      pct = wp["pct"], lo = wp["lo"], hi = wp["hi"], n = wp["n"],
      stringsAsFactors = FALSE
    )
  }
}
prev_df <- do.call(rbind, prev_rows)
rownames(prev_df) <- NULL

# ── 3. Logistic regression + King simulation ─────────────────
cat("Running logistic regressions (zelig2)...\n")

NUM_SIMS <- 1000
covariates_str <- "female + age_cat + race_f + english_prof + good_grades"

run_one_outcome <- function(outcome_col, label) {
  fml <- as.formula(paste(outcome_col, "~ racism_f +", covariates_str))
  sub <- df[complete.cases(df[, c(outcome_col, "racism_f", "weight",
                                   "female", "age_cat", "race_f",
                                   "english_prof", "good_grades")]), ]
  sub <- sub[sub$weight > 0, ]

  if (nrow(sub) < 50 || sum(sub[[outcome_col]]) < 5) {
    cat(sprintf("  SKIP %s (n=%d, events=%d)\n", label, nrow(sub),
                sum(sub[[outcome_col]])))
    return(NULL)
  }

  z <- tryCatch(
    zelig2(fml, model = "logit", data = sub, weights = sub$weight, num = NUM_SIMS),
    error = function(e) { cat(sprintf("  ERROR %s: %s\n", label, e$message)); NULL }
  )
  if (is.null(z)) return(NULL)

  results <- list()
  for (lev in c("Rarely", "Sometimes", "Most", "Always")) {
    z2 <- setx(z, racism_f = "Never", fn = "mean")
    z2 <- setx1(z2, racism_f = lev, fn = "mean")
    z2 <- sim(z2)
    qi <- zelig2_qi_to_df(z2)

    rr_draws <- qi$rr
    rr_draws <- rr_draws[is.finite(rr_draws)]

    if (length(rr_draws) < 10) {
      results[[lev]] <- data.frame(
        outcome = label, outcome_var = outcome_col, racism_level = lev,
        rr = NA, rr_lo = NA, rr_hi = NA, n = nrow(sub),
        stringsAsFactors = FALSE
      )
    } else {
      rr_q <- quantile(rr_draws, c(0.025, 0.5, 0.975))
      results[[lev]] <- data.frame(
        outcome = label, outcome_var = outcome_col, racism_level = lev,
        rr = rr_q[2], rr_lo = rr_q[1], rr_hi = rr_q[3], n = nrow(sub),
        stringsAsFactors = FALSE
      )
    }
  }
  do.call(rbind, results)
}

# Run for all outcomes
pr_rows <- list()
for (od in outcome_defs) {
  cat(sprintf("  %s...\n", od$label))
  res <- run_one_outcome(paste0(od$qn, "_bin"), od$label)
  if (!is.null(res)) pr_rows[[length(pr_rows) + 1]] <- res
}
pr_df <- do.call(rbind, pr_rows)
rownames(pr_df) <- NULL

# ── 4. Race/ethnicity–stratified analysis (all outcomes, all groups) ──
cat("Running race-stratified analyses...\n")

race_covs_str <- "female + age_cat + english_prof + good_grades"
full_strat_rows <- list()

for (grp in c("Non-White", "White", "Black", "Hispanic", "Asian",
              "AI/AN", "NH/PI", "Multiple")) {
  if (grp == "Non-White") {
    sub_grp <- df[!is.na(df$race_cat) & df$race_cat != "White", ]
    covs <- paste0(race_covs_str, " + race_f")
  } else {
    sub_grp <- df[!is.na(df$race_cat) & df$race_cat == grp, ]
    covs <- race_covs_str
  }

  for (od in outcome_defs) {
    outcome_col <- paste0(od$qn, "_bin")
    fml <- as.formula(paste(outcome_col, "~ racism_f +", covs))

    base_vars <- c(outcome_col, "racism_f", "weight", "female", "age_cat",
                   "english_prof", "good_grades")
    if (grp == "Non-White") base_vars <- c(base_vars, "race_f")

    sub2 <- sub_grp[complete.cases(sub_grp[, base_vars]), ]
    sub2 <- sub2[sub2$weight > 0, ]
    if (grp == "Non-White") sub2$race_f <- droplevels(sub2$race_f)

    if (nrow(sub2) < 50 || sum(sub2[[outcome_col]], na.rm = TRUE) < 5) {
      cat(sprintf("  SKIP %s / %s (n=%d)\n", grp, od$label, nrow(sub2)))
      for (lev in c("Rarely", "Sometimes", "Most", "Always")) {
        full_strat_rows[[length(full_strat_rows) + 1]] <- data.frame(
          group = grp, outcome = od$label, outcome_var = outcome_col,
          racism_level = lev, rr = NA, rr_lo = NA, rr_hi = NA, n = nrow(sub2),
          stringsAsFactors = FALSE
        )
      }
      next
    }

    z <- tryCatch(
      zelig2(fml, model = "logit", data = sub2, weights = sub2$weight, num = NUM_SIMS),
      error = function(e) {
        cat(sprintf("  ERROR %s / %s: %s\n", grp, od$label, e$message)); NULL
      }
    )
    if (is.null(z)) {
      for (lev in c("Rarely", "Sometimes", "Most", "Always")) {
        full_strat_rows[[length(full_strat_rows) + 1]] <- data.frame(
          group = grp, outcome = od$label, outcome_var = outcome_col,
          racism_level = lev, rr = NA, rr_lo = NA, rr_hi = NA, n = nrow(sub2),
          stringsAsFactors = FALSE
        )
      }
      next
    }

    for (lev in c("Rarely", "Sometimes", "Most", "Always")) {
      sim_res <- tryCatch({
        z2 <- setx(z, racism_f = "Never", fn = "mean")
        z2 <- setx1(z2, racism_f = lev, fn = "mean")
        z2 <- sim(z2)
        qi <- zelig2_qi_to_df(z2)
        rr_draws <- qi$rr[is.finite(qi$rr)]
        if (length(rr_draws) >= 10) {
          rr_q <- quantile(rr_draws, c(0.025, 0.5, 0.975))
          data.frame(group = grp, outcome = od$label, outcome_var = outcome_col,
                     racism_level = lev, rr = rr_q[2], rr_lo = rr_q[1],
                     rr_hi = rr_q[3], n = nrow(sub2), stringsAsFactors = FALSE)
        } else {
          data.frame(group = grp, outcome = od$label, outcome_var = outcome_col,
                     racism_level = lev, rr = NA, rr_lo = NA, rr_hi = NA,
                     n = nrow(sub2), stringsAsFactors = FALSE)
        }
      }, error = function(e) {
        cat(sprintf("    SIM ERROR %s / %s / %s: %s\n", grp, od$label, lev, e$message))
        data.frame(group = grp, outcome = od$label, outcome_var = outcome_col,
                   racism_level = lev, rr = NA, rr_lo = NA, rr_hi = NA,
                   n = nrow(sub2), stringsAsFactors = FALSE)
      })
      full_strat_rows[[length(full_strat_rows) + 1]] <- sim_res
    }
    cat(sprintf("  %s / %s done (n=%d)\n", grp, od$label, nrow(sub2)))
  }
}
full_strat_df <- do.call(rbind, full_strat_rows)
rownames(full_strat_df) <- NULL

# Backward-compatible extract for inline text
race_df <- full_strat_df[full_strat_df$outcome == "Bullied at school" &
                         full_strat_df$racism_level %in% c("Sometimes", "Always") &
                         full_strat_df$group %in% c("White", "Black", "Hispanic",
                                                    "Asian", "Multiple"), ]
race_df <- data.frame(race = race_df$group, outcome = race_df$outcome,
                      racism_level = race_df$racism_level,
                      rr = race_df$rr, rr_lo = race_df$rr_lo, rr_hi = race_df$rr_hi,
                      n = race_df$n, stringsAsFactors = FALSE)
rownames(race_df) <- NULL

# ── 4b. Collapsed-group analysis (3-level racism) ──────────────
cat("Running collapsed-group analyses (Never vs Rarely/Sometimes vs Most/Always)...\n")

df$racism_3 <- factor(
  ifelse(df$racism_f == "Never", "Never",
  ifelse(df$racism_f %in% c("Rarely", "Sometimes"), "Rarely/Sometimes",
  ifelse(df$racism_f %in% c("Most", "Always"), "Most/Always", NA))),
  levels = c("Never", "Rarely/Sometimes", "Most/Always")
)

collapsed_rows <- list()
for (grp in c("All", "Non-White", "White", "Black", "Hispanic", "Asian",
              "AI/AN", "NH/PI", "Multiple")) {
  if (grp == "All") {
    sub_grp <- df
    covs <- covariates_str  # includes race_f
  } else if (grp == "Non-White") {
    sub_grp <- df[!is.na(df$race_cat) & df$race_cat != "White", ]
    covs <- paste0(race_covs_str, " + race_f")
  } else {
    sub_grp <- df[!is.na(df$race_cat) & df$race_cat == grp, ]
    covs <- race_covs_str
  }

  for (od in outcome_defs) {
    outcome_col <- paste0(od$qn, "_bin")
    fml <- as.formula(paste(outcome_col, "~ racism_3 +", covs))

    base_vars <- c(outcome_col, "racism_3", "weight", "female", "age_cat",
                   "english_prof", "good_grades")
    if (grp %in% c("All", "Non-White")) base_vars <- c(base_vars, "race_f")

    sub2 <- sub_grp[complete.cases(sub_grp[, base_vars]), ]
    sub2 <- sub2[sub2$weight > 0, ]
    if (grp == "Non-White") sub2$race_f <- droplevels(sub2$race_f)

    if (nrow(sub2) < 50 || sum(sub2[[outcome_col]], na.rm = TRUE) < 5) {
      cat(sprintf("  SKIP %s / %s (collapsed, n=%d)\n", grp, od$label, nrow(sub2)))
      for (lev in c("Rarely/Sometimes", "Most/Always")) {
        collapsed_rows[[length(collapsed_rows) + 1]] <- data.frame(
          group = grp, outcome = od$label, outcome_var = outcome_col,
          racism_level = lev, rr = NA, rr_lo = NA, rr_hi = NA, n = nrow(sub2),
          stringsAsFactors = FALSE
        )
      }
      next
    }

    z <- tryCatch(
      zelig2(fml, model = "logit", data = sub2, weights = sub2$weight, num = NUM_SIMS),
      error = function(e) {
        cat(sprintf("  ERROR %s / %s (collapsed): %s\n", grp, od$label, e$message)); NULL
      }
    )
    if (is.null(z)) {
      for (lev in c("Rarely/Sometimes", "Most/Always")) {
        collapsed_rows[[length(collapsed_rows) + 1]] <- data.frame(
          group = grp, outcome = od$label, outcome_var = outcome_col,
          racism_level = lev, rr = NA, rr_lo = NA, rr_hi = NA, n = nrow(sub2),
          stringsAsFactors = FALSE
        )
      }
      next
    }

    for (lev in c("Rarely/Sometimes", "Most/Always")) {
      sim_res <- tryCatch({
        z2 <- setx(z, racism_3 = "Never", fn = "mean")
        z2 <- setx1(z2, racism_3 = lev, fn = "mean")
        z2 <- sim(z2)
        qi <- zelig2_qi_to_df(z2)
        rr_draws <- qi$rr[is.finite(qi$rr)]
        if (length(rr_draws) >= 10) {
          rr_q <- quantile(rr_draws, c(0.025, 0.5, 0.975))
          data.frame(group = grp, outcome = od$label, outcome_var = outcome_col,
                     racism_level = lev, rr = rr_q[2], rr_lo = rr_q[1],
                     rr_hi = rr_q[3], n = nrow(sub2), stringsAsFactors = FALSE)
        } else {
          data.frame(group = grp, outcome = od$label, outcome_var = outcome_col,
                     racism_level = lev, rr = NA, rr_lo = NA, rr_hi = NA,
                     n = nrow(sub2), stringsAsFactors = FALSE)
        }
      }, error = function(e) {
        cat(sprintf("    SIM ERROR %s / %s / %s (collapsed): %s\n", grp, od$label, lev, e$message))
        data.frame(group = grp, outcome = od$label, outcome_var = outcome_col,
                   racism_level = lev, rr = NA, rr_lo = NA, rr_hi = NA,
                   n = nrow(sub2), stringsAsFactors = FALSE)
      })
      collapsed_rows[[length(collapsed_rows) + 1]] <- sim_res
    }
    cat(sprintf("  %s / %s done (collapsed)\n", grp, od$label))
  }
}
collapsed_df <- do.call(rbind, collapsed_rows)
rownames(collapsed_df) <- NULL

# ── 5. Save results ──────────────────────────────────────────
write.csv(prev_df, file.path(DATA_DIR, "prevalences.csv"), row.names = FALSE)
write.csv(pr_df, file.path(DATA_DIR, "risk_ratios.csv"), row.names = FALSE)
write.csv(race_df, file.path(DATA_DIR, "race_stratified_rr.csv"), row.names = FALSE)
write.csv(full_strat_df, file.path(DATA_DIR, "stratified_rr.csv"), row.names = FALSE)
write.csv(collapsed_df, file.path(DATA_DIR, "collapsed_rr.csv"), row.names = FALSE)

# Also save racism prevalence by race for text
any_rows <- list()
for (race in c("Overall", "Asian", "Black", "Multiple", "Hispanic", "White")) {
  if (race == "Overall") {
    sub <- df[!is.na(df$racism_f), ]
  } else {
    sub <- df[!is.na(df$racism_f) & !is.na(df$race_cat) & df$race_cat == race, ]
  }
  sub$any_racism <- as.numeric(sub$racism_f != "Never")
  sub$most_always <- as.numeric(sub$racism_f %in% c("Most", "Always"))
  wp_any <- wprev(sub$any_racism, sub$weight)
  wp_ma  <- wprev(sub$most_always, sub$weight)
  any_rows[[length(any_rows) + 1]] <- data.frame(
    group = race, n = nrow(sub),
    any_pct = wp_any["pct"], any_lo = wp_any["lo"], any_hi = wp_any["hi"],
    ma_pct = wp_ma["pct"],
    stringsAsFactors = FALSE
  )
}
racism_prev_df <- do.call(rbind, any_rows)
rownames(racism_prev_df) <- NULL
write.csv(racism_prev_df, file.path(DATA_DIR, "racism_prevalence.csv"), row.names = FALSE)

cat(sprintf("\nResults saved to %s/\n", DATA_DIR))
cat(sprintf("  prevalences.csv:       %d rows\n", nrow(prev_df)))
cat(sprintf("  risk_ratios.csv:       %d rows\n", nrow(pr_df)))
cat(sprintf("  race_stratified_rr.csv: %d rows\n", nrow(race_df)))
cat(sprintf("  stratified_rr.csv:     %d rows\n", nrow(full_strat_df)))
cat(sprintf("  collapsed_rr.csv:     %d rows\n", nrow(collapsed_df)))
cat(sprintf("  racism_prevalence.csv: %d rows\n", nrow(racism_prev_df)))
cat("Done.\n")
