set -x
set -e

./scripts/training.sh
./scripts/training.sh loss_normalization_and_gradient_clipping
./scripts/training.sh low_threshold_instance_merging
./scripts/training.sh min_area_fraction
./scripts/training.sh mosaics
./scripts/training.sh same_learning_rates