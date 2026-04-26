suppressPackageStartupMessages(library(nhlscraper))
suppressPackageStartupMessages(library(arrow))

args <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("^--file=", args, value = TRUE)
script_dir <- if (length(script_arg) == 1) {
  dirname(normalizePath(sub("^--file=", "", script_arg)))
} else {
  getwd()
}

dir.create(file.path(script_dir, "data"), recursive = TRUE, showWarnings = FALSE)

load_regular_pbp <- function(season_id) {
  pbp <- gc_play_by_plays(season_id)
  pbp[pbp$gameTypeId == 2, ]
}

bind_rows_union <- function(frames) {
  all_cols <- unique(unlist(lapply(frames, names)))
  aligned <- lapply(frames, function(df) {
    missing_cols <- setdiff(all_cols, names(df))
    if (length(missing_cols) > 0) {
      for (col in missing_cols) {
        df[[col]] <- NA
      }
    }
    df[, all_cols, drop = FALSE]
  })
  do.call(rbind, aligned)
}

train_seasons <- c(20142015, 20152016, 20162017, 20172018)
test_seasons <- c(20182019)

train <- bind_rows_union(lapply(train_seasons, load_regular_pbp))
test <- bind_rows_union(lapply(test_seasons, load_regular_pbp))

write_parquet(train, file.path(script_dir, "data", "train.parquet"))
write_parquet(test, file.path(script_dir, "data", "test.parquet"))
