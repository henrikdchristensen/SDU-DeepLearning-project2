# Readme
All the results are stored in the `notebook.ipynb`.

## Create conda environment
```bash
conda create -n deep python=3.12
conda activate deep
pip install -r requirements.txt
```

## Without CUDA
Remove the line `--extra-index-url` specified in `requirements.txt` and remove `*-cuXXX` at the end.

## How to install CUDA
Note that step 3 is only neccessary if you are using another version than the one specified in requirements.txt.

1. Under Compute Platform, check the latest supported CUDA toolkit version for PyTorch: https://pytorch.org/get-started/locally/.
2. Download corresponding CUDA toolkit version from https://developer.nvidia.com/cuda-toolkit-archive.
3. (Optional - manual install cuda support into pip enviroment) Select your OS, package=pip, and CUDA version, and run the pip install command showing on the PyTorch webpage (https://pytorch.org/get-started/locally/). Remember to activate conda enviroment. The command could look like:
```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## Export notebook to LaTeX
```bash
jupyter nbconvert --to latex notebook.ipynb
```
Change `\documentclass[11pt]{article}` with `\documentclass[8pt]{extarticle}` and export to pdf.

## Export code to pdf
```bash
python export_code_to_pdf.py
```

## Format code
```bash
black --line-length 100 . 
```