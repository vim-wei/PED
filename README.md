## Improving Zero-Shot Generalization for CLIP with Prompt Ensemble self-Distillation

## Installation
You can prepare the environment as follows:

* Setup conda environment.
```bash
# Create a conda environment
conda create -n ped python=3.7

# Activate the environment
conda activate ped

# Install dependencies
pip install -r requirements.txt

# Install torch (version >= 1.7.1) and torchvision
# Please make sure you have installed the gpu version due to the speed.
# For example:
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

* Install [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch) library.
```bash
# Instructions borrowed from https://github.com/KaiyangZhou/Dassl.pytorch#installation

# Clone this repo
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
cd ..
```

Follow [DATASETS.md](DATASETS.md) to install the datasets.

## Running the Code

```bash
bash run_and_eval.sh
```

## Acknowledgement

This repo benefits from [CLIP](https://github.com/openai/CLIP), [CoOp](https://github.com/KaiyangZhou/Dassl.pytorch) and [TCP](https://github.com/htyao89/Textual-based_Class-aware_prompt_tuning). Thanks for their wonderful works.
