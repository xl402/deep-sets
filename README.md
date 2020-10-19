# Deep-Sets
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/xl402/deep-sets/deep-sets-transformer)

A modified implementation of the [Set Transformer](http://proceedings.mlr.press/v97/lee19d/lee19d.pdf) using Tensorflow and Keras.

### Problem Definition
A function <img src="https://render.githubusercontent.com/render/math?math=f: X^n \rightarrow Y^n"> is permutation equivariant iff for any permutation <img src="https://render.githubusercontent.com/render/math?math=\pi">: <img src="https://render.githubusercontent.com/render/math?math=f(\pi x) = \pi f(x)">

Similarly, a function is permutation equivariant if: <img src="https://render.githubusercontent.com/render/math?math=f(\pi x) = f(x)">

Neural networks tend to satisfy neither of these conditions by construction. This repo implements a version of the [Set Transformer](https://arxiv.org/abs/1810.00825).

### Initial Setup
#### Install dependencies
This repo requires Python 3.7 and above. Install requirements by running:
```
pip install -r requirements.txt
```
Then expoert `src` to path:
```
export PYTHONPATH=PATH_TO_REPO/src
```
To test the scripts, run `pytest` in the root directory
