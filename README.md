# Stable, accurate and efficient deep neural networks for inverse problems with analysis-sparse models

This repository contains the numerical experiments of the paper [Stable, accurate and efficient deep neural networks for inverse problems with analysis-sparse models](https://arxiv.org/abs/2203.00804), authored by Maksym Neyra-Nesterenko and Ben Adcock (2022).

## Requirements

The experiments are written in [Python](https://www.python.org/downloads/) and can be run on any Linux-based distribution. 

To run the experiments without issues, these were run with Python 3.9 and using

| Package | Version |
| ------- | ------- |
| `matplotlib` | 3.5.1 |
| `numpy` | 1.22.0 |
| `Pillow` | 9.0.0 |
| `scipy` | 1.7.3 |
| `seaborn` | 0.11.2 |
| `torch` | 1.10.1 |

We recommend using these versions or later versions.

## Running the experiments

To run any of the experiments under `demos/`, we suggest you use a Python [virtual environment](https://docs.python.org/3.9/library/venv.html) to set things up.

Proceeding, first create the virtual environment and source it:

```shell
$ mkdir env
$ python3 -m venv env
$ source env/bin/activate
```

Afterwards, clone the repository and then install the `nestanet` package defined in `setup.py`. This will install the requirements above as dependencies.

```shell
(env) $ git clone https://github.com/mneyrane/AS-NESTA-net.git
(env) $ cd AS-NESTA-net
(env) $ pip install -e .
```
Now you can run any of the code found in `demos/`!

## Issues

You can post questions, requests, and bugs in [Issues](https://github.com/mneyrane/AS-NESTA-net/issues).

## Acknowledgements

The unrolled NESTA implementation is adapted from the code for [FIRENET](https://github.com/Comp-Foundations-and-Barriers-of-AI/firenet).
