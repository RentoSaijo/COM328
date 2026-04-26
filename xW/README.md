# xW Project

## Overview

This directory contains the COM328 course project on hockey expected wins (`xW`). The main research question is:

**Can we predict which team should have won an NHL game from the full event sequence, instead of relying only on a cumulative summary such as xG differential?**

The project uses NHL regular-season play-by-play data from `2014-15` through `2018-19`, builds a fixed-width event-sequence representation for each team-game row, studies its structure with PCA/UMAP/KMeans, and compares two supervised models:

- `XGBoost + Optuna` on a flattened wide matrix
- a `keras` temporal convolutional network on the ordered event sequence

The submission files are:

- [scrape.R](/Users/rsai_91/Desktop/Academic/Spring2026/COM328/xW/scrape.R): collects and stores the raw play-by-play data
- [code.ipynb](/Users/rsai_91/Desktop/Academic/Spring2026/COM328/xW/code.ipynb): full self-contained notebook with code, questions, outputs, and answers
- [slides.pptx](/Users/rsai_91/Desktop/Academic/Spring2026/COM328/xW/slides.pptx): 5-slide project summary deck

## Research Question

The standard hockey-analytics shortcut is to summarize a game into a few aggregate statistics, then infer which team deserved to win. This project tests a stricter idea:

- treat the game as an ordered event list
- keep the local context of each event
- ask whether a sequence-aware model can recover the deserved winner more effectively than a flattened baseline

This makes the project a better fit for the course requirements because the raw source data is very large, the engineered matrix exceeds `1000` dimensions, and the final modeling stage uses architectures beyond the course's regular material.

## Data Source And Scraping

### Source

The raw data comes from the CRAN R package `nhlscraper`, using:

- `nhlscraper::gc_play_by_plays(20142015)`
- `nhlscraper::gc_play_by_plays(20152016)`
- `nhlscraper::gc_play_by_plays(20162017)`
- `nhlscraper::gc_play_by_plays(20172018)`
- `nhlscraper::gc_play_by_plays(20182019)`

Only regular-season games are kept with `typeId == 2`.

### Split

- training seasons: `2014-15`, `2015-16`, `2016-17`, `2017-18`
- held-out test season: `2018-19`

### Storage

The scraping step is intentionally separated into `scrape.R` and saves only:

- `/Users/rsai_91/Desktop/Academic/Spring2026/COM328/xW/data/train.parquet`
- `/Users/rsai_91/Desktop/Academic/Spring2026/COM328/xW/data/test.parquet`

The parquet files are much smaller than the earlier CSV version:

- `train.parquet`: about `83.72 MB`
- `test.parquet`: about `21.54 MB`

## Data Cleaning And Transformation

### Retained events

The model keeps the event types requested for the project-style representation:

- `faceoff`
- `hit`
- `shot-on-goal`
- `giveaway`
- `missed-shot`
- `blocked-shot`
- `goal`, recoded to `shot-on-goal`
- `takeaway`

`goal` is converted to `shot-on-goal` so the predictor matrix does not leak the final game outcome into the inputs.

### Situation handling

The raw `situationCode` is cleaned as a character variable and zero-padded when needed. For example:

- `651` becomes `0651`

The original NHL ordering is:

- `(away goalie, away skaters, home skaters, home goalie)`

The project reorders it relative to the event owner:

- `(event owner goalie, event owner skaters, opponent skaters, opponent goalie)`

That makes the situation columns team-relative rather than home/away-relative.

### Team-row construction

Each game is duplicated into two rows:

- one row from the home-team perspective
- one row from the away-team perspective

Both rows see the exact same retained event list. The difference is the `event_i_isOwner` flag, which tells us whether the team represented by that row caused that event.

The row-level response is:

- `won = 1` if that team won the game
- `won = 0` otherwise

### Fixed-width sequence design

Games have different numbers of retained events, so the notebook builds a consistent wide matrix by keeping an evenly spaced `180`-event subsequence across each full retained game. This choice was made for two reasons:

- it satisfies the course requirement of more than `1000` dimensions
- it preserves early, middle, and late-game context better than simply taking the first `180` events

Each retained event contributes six columns:

- `event_i_isOwner`
- `event_i_situation`
- `event_i_time`
- `event_i_xNorm`
- `event_i_yNorm`
- `event_i_type`

So the event-level predictor count is:

- `180 x 6 = 1080`

The final widened matrices have:

- training shape: `9886 x 1087`
- test shape: `2542 x 1087`

The extra 7 columns are the identifying/outcome columns such as game id, team id, and `won`.

## Why The `180`-Event Cutoff Was Chosen

The minimum retained-event count would have been too small to clear the dimensionality requirement. The project compared multiple cutoffs and selected `180` because it gave more than `1000` predictors while dropping very little data.

At `180` events:

- predictor dimension: `1080`
- training games kept: `4943 / 4961`
- training games dropped: `18`
- test games kept: `1271 / 1271`
- test games dropped: `0`

This is the cutoff used throughout the notebook and slides.

## Notebook Structure

The notebook is intentionally self-contained:

- all helper functions are defined at the beginning of the notebook
- the helper section is commented in the same style as the rest of the assignment code
- all later sections reuse cached outputs in `xW/cache` and `xW/outputs` when available

The notebook follows the class convention from the homework/quiz files:

- `Question:` markdown cell
- code/output cell
- `Answer:` markdown cell with multi-part bullets

## Unsupervised Analysis

The widened training matrix is standardized, then analyzed with:

- `PCA`
- `UMAP`
- `KMeans` on the 2D UMAP embedding

### Main EDA results

- first two PCA variance ratios: `0.104125` and `0.027617`
- principal components needed for 90% variance: `673`
- best UMAP + KMeans solution: `k = 2`
- best silhouette score: `0.499453`
- cluster sizes: `4139` and `5747`
- cluster win rates: `0.5006` and `0.4996`

### Interpretation

The event matrix clearly has structure, but the structure is not a clean winner-vs-loser split. The clusters appear to reflect broad game-style or flow differences more than direct outcome labels, which is consistent with the nearly identical win rates in the two clusters.

That is useful for the project:

- it shows the widened sequence representation is not random noise
- it also shows that winning is still a difficult target because the manifold is noisy and overlapping

## Supervised Models

### 1. XGBoost + Optuna

The baseline supervised model is a flattened tree model:

- input: the full `1080`-feature wide table
- tuning method: `optuna`
- objective: binary logistic classification
- validation metric: AUC

Best validation AUC:

- `0.886697`

Held-out `2018-19` test metrics:

- accuracy: `0.7714`
- precision: `0.7721`
- recall: `0.7703`
- F1: `0.7712`
- AUC: `0.8893`
- log-loss: `0.4217`

### 2. Temporal CNN

The second model is a sequence-aware neural network:

- numeric sequence channels: `isOwner`, `time`, `xNorm`, `yNorm`
- categorical sequence channels: `type`, `situation`
- categorical handling: learned embeddings
- sequence model: stacked `Conv1D` layers plus global max pooling

Best validation AUC:

- `0.918209`

Best epoch:

- `10`

Parameter count:

- `9081`

Held-out `2018-19` test metrics:

- accuracy: `0.8131`
- precision: `0.7632`
- recall: `0.9079`
- F1: `0.8293`
- AUC: `0.9290`
- log-loss: `0.3416`

## Model Interpretation

### What XGBoost learned

The top XGBoost gain features map mostly to late-sequence fields:

- `event_173_isOwner`
- `event_175_isOwner`
- `event_177_isOwner`
- `event_178_isOwner`
- `event_176_type`
- late `event_158` to `event_167` situation fields

This means the flattened tree model relies heavily on:

- who owned the later retained events
- what the late-game strength state looked like

When the notebook averages the final 30 retained `isOwner` indicators, the results are:

- actual loss rows: `0.5343`
- actual win rows: `0.4657`

That is a meaningful hockey interpretation. It suggests a **score-effects** pattern:

- teams that are trailing often drive more late events because they are pushing to tie the game
- teams that are leading can be more passive or defensive late

So the flattened baseline learns a useful but somewhat narrow end-of-game signal.

### What the temporal CNN adds

The temporal CNN improves on that baseline because it can learn short-range event patterns directly from the ordered sequence. Instead of seeing `event_176_type` and `event_177_isOwner` as isolated columns, it can learn interactions such as:

- who owned consecutive events
- whether those events occurred under favorable or unfavorable situation codes
- where they happened
- when they occurred in local sequence context

Its test recall is much higher:

- XGBoost recall: `0.7703`
- temporal CNN recall: `0.9079`

Its confusion matrix also shows:

- XGBoost true loss rows called wins: `289`
- temporal CNN true loss rows called wins: `358`

In a standard classifier this would just be a false-positive cost. In this project, those rows are especially interesting because they are candidate games where the event flow looked more like a winning performance than the final scoreboard did.

That is why the temporal CNN is the better xW-style model candidate.

## Main Conclusions

1. The sequence representation works. The project generalizes to a held-out season and produces strong out-of-sample performance.
2. Unsupervised structure exists, but it is not equivalent to direct outcome separation. The matrix captures style/flow information as well as win/loss information.
3. A sequence-aware neural model is the best fit for the project goal. The temporal CNN outperformed the tuned flattened baseline on every major test metric.
4. The disagreement cases are the most interesting hockey cases. Rows predicted as wins that actually lost are natural candidates for “should-have-won” interpretations.

## Limitations

- Only regular-season games from `2014-15` through `2018-19` are used.
- The project does not add roster strength, goalie talent, score state history, or special-team context beyond the encoded event situations.
- The fixed-width representation keeps `180` evenly spaced retained events, so some within-game detail is intentionally compressed.
- The current notebook predicts row-level wins rather than turning those outputs into a final game-level xW reporting metric.

## Recommended Next Step

The clearest extension is:

- convert the team-row win probabilities into a game-level expected-win disagreement score

That would let the project answer a more hockey-native question such as:

- Which losing teams had the strongest event-sequence case that they should have won?

## Reproducing The Project

From `/Users/rsai_91/Desktop/Academic/Spring2026/COM328/xW`:

1. Run the scraper:

```bash
Rscript scrape.R
```

2. Execute the notebook in place:

```bash
jupyter execute code.ipynb --inplace
```

3. Rebuild the slide deck if needed:

```bash
"/Users/rsai_91/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node" presentation_workspace/src/deck.mjs
```

## Supporting Files

- `cache/`: cached widened matrices
- `outputs/analysis_results.json`: saved numeric results used by the notebook
- `outputs/pca_2_embedding.npy`: PCA embedding
- `outputs/umap_2_embedding.npy`: UMAP embedding
- `presentation_workspace/`: editable slide source, previews, and export workspace
