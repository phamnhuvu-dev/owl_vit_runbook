set -x
set -e

export PYTHONPATH="${PYTHONPATH}:/big_vision/"

python -m runbook.scenic.projects.owl_vit.evaluator \
  --alsologtostderr=true \
  --platform=gpu \
  --config=clip_b32 \
  --checkpoint_path=gs://scenic-bucket/owl_vit/checkpoints/clip_vit_b32_b0203fc \
  --annotations_path=annotations/instances_val2017.json \
  --tfds_name=coco/2017 \
  --data_format=coco \
  --output_dir=/tmp/evaluator