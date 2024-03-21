set -x
set -e

export PYTHONPATH="${PYTHONPATH}:/big_vision/"

XLA_PYTHON_CLIENT_PREALLOCATE=false python -m scenic_repo.scenic.projects.owl_vit.main \
  --alsologtostderr=true \
  --workdir=/tmp/training \
  --config=scenic_repo/scenic/projects/owl_vit/configs/clip_b32_finetune.py
