def update_file(file_path, old, new):
    with open(file_path, 'r') as file:
        content = file.read()
    content = content.replace(old, new)
    with open(file_path, 'w') as file:
        file.write(content)

update_file('runbook/scenic/projects/owl_vit/evaluator.py', 'set_cache_dir', 'initialize_cache')
update_file('ott/src/ott/initializers/nn/initializers.py', 'rng: jax.random.PRNGKeyArray = jax.random.PRNGKey(0),', 'rng: jax.Array = jax.random.PRNGKey(0),')