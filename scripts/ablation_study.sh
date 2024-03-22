set -x
set -e


ablation_study() {
    export PYTHONPATH="${PYTHONPATH}:/big_vision/"
    ./scripts/training.sh
    ./scripts/training.sh loss_normalization_and_gradient_clipping
    ./scripts/training.sh low_threshold_instance_merging
    ./scripts/training.sh min_area_fraction
    ./scripts/training.sh mosaics
    ./scripts/training.sh same_learning_rates
}

while [ true ]
do
  ablation_study || echo "Ablation study failed" >> failed_ablation_study.txt
done