set -x
set -e


# Remove all files and folders in runbook/ except for .ipynb files
mv runbook/*.ipynb . || true

cd runbook
rm -rf *
rm -rf .config
rm -rf .git

# Install additional dependencies
apt update && apt install -y libgl1-mesa-glx
pip install cython
pip install apache-beam
pip install tensorflow-datasets

# Install owl_vit dependencies in scenic
git clone https://github.com/google-research/scenic.git .
pip install -q .
pip install -r ./scenic/projects/owl_vit/requirements.txt
pip install scikit-image
pip install git+https://github.com/openai/CLIP.git

# Install big_vision, which is needed for the mask head:
rm -r /big_vision || true
mkdir /big_vision
git clone https://github.com/google-research/big_vision.git /big_vision
pip install -r /big_vision/big_vision/requirements.txt

rm -rf .git

cd ..
mv *.ipynb runbook/ || true

git clone --depth 1 --branch 0.3.1 https://github.com/ott-jax/ott.git || true
# Run the update_unable_run_code.py script to remove the code cells that are not able to run
python update_unable_run_code.py
pip install -e ott/
pip install runbook/
pip install numpy==1.24.4
pip install pycocotools==2.0.6
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html