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
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -q .
pip install -r ./scenic/projects/owl_vit/requirements.txt
pip install scikit-image
pip install pycocotools==2.0.6
pip install ott-jax==0.3.1
pip install git+https://github.com/openai/CLIP.git
pip install jax==0.4.23
pip install jaxlib@https://storage.googleapis.com/jax-releases/cuda12/jaxlib-0.4.23+cuda12.cudnn89-cp310-cp310-manylinux2014_x86_64.whl#sha256=8e42000672599e7ec0ea7f551acfcc95dcdd0e22b05a1d1f12f97b56a9fce4a8


# Install big_vision, which is needed for the mask head:
rm -r /big_vision || true
mkdir /big_vision
git clone https://github.com/google-research/big_vision.git /big_vision
pip install -r /big_vision/big_vision/requirements.txt
pip install numpy==1.24.4

rm -rf .git

cd ..
mv *.ipynb runbook/ || true