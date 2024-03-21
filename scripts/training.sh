set -x
set -e

export PYTHONPATH="${PYTHONPATH}:/big_vision/"

ABLATION_STUDY=$1
WORKDIR=''

if [ $ABLATION_STUDY == 'loss_normalization_and_gradient_clipping' ]; then
elif [ $ABLATION_STUDY == 'low_threshold_instance_merging' ]; then
  WORKDIR='/tmp/training_low_threshold_instance_merging'
  CONFIG='ablation_study/clip_b32_low_threshold_instance_merging.py'
elif [ $ABLATION_STUDY == 'same_learning_rates' ]; then
  WORKDIR='/tmp/training_same_learning_rates'
  CONFIG='ablation_study/clip_b32_same_learning_rates.py'
else
  WORKDIR='/tmp/training'
  CONFIG='scenic_repo/scenic/projects/owl_vit/configs/clip_b32.py'
  echo "Default"
fi

XLA_PYTHON_CLIENT_PREALLOCATE=false python -m scenic_repo.scenic.projects.owl_vit.main \
  --alsologtostderr=true \
  --workdir=$WORKDIR \
  --config=$CONFIG
  