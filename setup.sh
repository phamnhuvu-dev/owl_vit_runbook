set -x
set -e

pip install bokeh
rm -r colabtools || true
git clone https://github.com/googlecolab/colabtools.git
cd colabtools
python setup.py install
cd ..

# Install additional dependencies
apt update && apt install -y libgl1-mesa-glx
pip install cython
pip install apache-beam
pip install tensorflow-datasets

# Install owl_vit dependencies in scenic
rm -r scenic_repo || true
git clone https://github.com/google-research/scenic.git scenic_repo/
pip install -r scenic_repo/scenic/projects/owl_vit/requirements.txt
pip install scikit-image
# pip install git+https://github.com/openai/CLIP.git

# Install big_vision, which is needed for the mask head:
rm -r /big_vision || true
mkdir /big_vision
git clone https://github.com/google-research/big_vision.git /big_vision
pip install -r /big_vision/big_vision/requirements.txt


git clone --depth 1 --branch 0.3.1 https://github.com/ott-jax/ott.git || true
# Run the update_unable_run_code.py script to remove the code cells that are not able to run
python update_unable_run_code.py
pip install scenic_repo/
pip install -e ott/
pip install numpy==1.24.4
pip install pycocotools==2.0.6
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip uninstall -y opencv-python