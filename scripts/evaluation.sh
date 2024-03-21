set -x
set -e

export PYTHONPATH="${PYTHONPATH}:/big_vision/"

CONFIG=$1
CHECKPOINT_PATH=''

if [ $CONFIG == 'clip_b32' ]; then
  CHECKPOINT_PATH='gs://scenic-bucket/owl_vit/checkpoints/clip_vit_b32_b0203fc'
elif [ $CONFIG == 'clip_b16' ]; then
  CHECKPOINT_PATH='gs://scenic-bucket/owl_vit/checkpoints/clip_vit_b16_6171dab'
elif [ $CONFIG == 'clip_l14' ]; then
  CHECKPOINT_PATH='gs://scenic-bucket/owl_vit/checkpoints/clip_vit_l14_d83d374'
elif [ $CONFIG == 'owl_v2_clip_b16' ]; then
  CHECKPOINT_PATH='gs://scenic-bucket/owl_vit/checkpoints/owl2-b16-960-st-ngrams-curated-ft-lvisbase-ens-cold-weight-05_209b65b'
elif [ $CONFIG == 'owl_v2_clip_l14' ]; then
  CHECKPOINT_PATH='gs://scenic-bucket/owl_vit/checkpoints/owl2-l14-1008-st-ngrams-ft-lvisbase-ens-cold-weight-04_8ca674c'
else
  echo "Invalid config"
  exit 1
fi

python -m scenic_repo.scenic.projects.owl_vit.evaluator \
  --alsologtostderr=true \
  --platform=gpu \
  --config=$CONFIG \
  --checkpoint_path=$CHECKPOINT_PATH \
  --annotations_path=annotations/instances_val2017.json \
  --tfds_name=coco/2017 \
  --data_format=coco \
  --output_dir=/tmp/evaluator

  # --checkpoint_path=gs://scenic-bucket/owl_vit/checkpoints/clip_vit_b32_b0203fc \
  # --checkpoint_path=/tmp/training/checkpoint_140000 \