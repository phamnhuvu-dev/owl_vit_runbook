def update_file(file_path, old, new):
    with open(file_path, 'r') as file:
        content = file.read()
    content = content.replace(old, new)
    with open(file_path, 'w') as file:
        file.write(content)

def append_file(file_path, new):
    with open(file_path, 'a') as file:
        file.write(new)

# Update unable run code
update_file('scenic_repo/scenic/projects/owl_vit/evaluator.py', 'set_cache_dir', 'initialize_cache')
update_file('ott/src/ott/initializers/nn/initializers.py', 'rng: jax.random.PRNGKeyArray = jax.random.PRNGKey(0),', 'rng: jax.Array = jax.random.PRNGKey(0),')

# Check if text_key != negative_text_labels
update_file('scenic_repo/scenic/projects/owl_vit/preprocessing/label_ops.py', 'if text_key in features:', 'if text_key != \'negative_text_labels\' and text_key in features:')

# Add DecodeCoco class to label_ops.py
decodeCocoCode = '''
@dataclasses.dataclass(frozen=True)
class DecodeCoco(image_ops.DecodeCocoExample):
  
  is_promptable: bool = True
  tfds_data_dir: Optional[str] = None

  def __call__(self, features: Features) -> Features:
    new_features = super().__call__(features)
    new_features[modalities.NEGATIVE_TEXT_LABELS] = tf.fill([1], PADDING_QUERY)
    return IntegerToTextLabels(
        tfds_name='coco', is_promptable=self.is_promptable)(new_features)
'''
append_file('scenic_repo/scenic/projects/owl_vit/preprocessing/label_ops.py', decodeCocoCode)
update_file('scenic_repo/scenic/projects/owl_vit/preprocessing/input_pipeline.py', 'DECODERS = {', 'DECODERS = {\n    \'coco:1.1.0\': label_ops.DecodeCoco,\n')
