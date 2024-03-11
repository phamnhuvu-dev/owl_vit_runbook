set -x
set -e

export PYTHONPATH="${PYTHONPATH}:/big_vision/"

XLA_PYTHON_CLIENT_PREALLOCATE=false python -m runbook.scenic.projects.owl_vit.main \
  --alsologtostderr=true \
  --workdir=/tmp/training/clip \
  --config=runbook/scenic/projects/owl_vit/configs/clip_b32_finetune.py