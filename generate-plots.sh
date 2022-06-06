#!/usr/bin/env sh

# Evaluate all models on the test data of all domains
scripts/run-loss.sh
# Use the results to generate the loss table
scripts/losstable.sh

# Generate training performance plots for all domains+models
scripts/training-plots.sh

# Display weight functions learned by SC2 TPC models
scripts/sc-filters.sh

# Evaluate models on SC2 domains @ different prediction timesteps
scripts/run-sc-dist.sh
# Plot the resulting bar charts
scripts/sc-dist.sh

# Display attention weighting by SC2 TPCA model
scripts/sc-attn.sh

# Display traffic error table
scripts/tr-table.sh

# Display traffic models' prediction over certain days
scripts/tr-day.sh

# Calculate weather models' dropout performance
scripts/we-ensemble.sh
