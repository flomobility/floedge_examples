# Flo Edge examples

This repo contains [Flo Edge](https://wiki.flomobility.com/) examples

## Usage
> All examples are witten in python

### Setup Python Environment
Install miniconda
```
wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
bash Miniconda3-latest-Linux-aarch64.sh
```
Restart terminal after setup and create a Python 3.8 environment
```
conda create -n pyenv python=3.8
conda activate pyenv
```

### Set up git lfs
For all the models to download automatically when this repo is clones, git lfs needs to be installed
```
sudo apt-get install git-lfs
git-lfs install
```

### Clone this repo
```
git clone https://github.com/flomobility/floedge_examples.git
cd floedge_examples/
```

### Install dependencies
Install all dependencies in the requirements.txt file
```
pip install -r requirements.txt
```

### Run an example
Either run the inference.py script of any example or refer to the example's README for more details
```
cd ai/yolo_v5/
python inference.py
```
Use ```--help``` while running the inference for more details
