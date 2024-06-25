# femo_alpha

<!---
[![Python](https://img.shields.io/pypi/pyversions/femo_alpha)](https://img.shields.io/pypi/pyversions/femo_alpha)
[![Pypi](https://img.shields.io/pypi/v/femo_alpha)](https://pypi.org/project/femo_alpha/)
[![Coveralls Badge][13]][14]
[![PyPI version][10]][11]
[![PyPI Monthly Downloads][12]][11]
-->

[![GitHub Actions Test Badge](https://github.com/LSDOlab/femo_alpha/actions/workflows/actions.yml/badge.svg)](https://github.com/femo_alpha/femo_alpha/actions)
[![Forks](https://img.shields.io/github/forks/LSDOlab/femo_alpha.svg)](https://github.com/LSDOlab/femo_alpha/network)
[![Issues](https://img.shields.io/github/issues/LSDOlab/femo_alpha.svg)](https://github.com/LSDOlab/femo_alpha/issues)

A general framework for using **F**inite **E**lement in PDE-constrained **M**ultidisciplinary **O**ptimization problems. It relies on [FEniCSx](https://fenicsproject.org/) to provide solutions and partial derivatives of the PDE residuals, and uses [CSDL_alpha](https://github.com/LSDOlab/CSDL_alpha) as the umbrella for coupling and mathematical modeling of the multidisciplinary optimization problem. 



# Installation

The minimal requirements to use **femo_alpha** for modeling and simulation are `FEniCSx` and `CSDL_alpha`. For modeling aircraft design applications, you may install [CADDEE_alpha](https://github.com/LSDOlab/CADDEE_alpha) to enable coupling with solvers of other disciplines; for optimization, you will also need [ModOpt](https://github.com/LSDOlab/modopt) on top of them for Python bindings of various state-of-the-art optimizers. 

## Installation instructions for users
It's recommended to use conda for installing the module and its dependencies.

- Create a conda environment for `femo` with a specific Python version (Python 3.9) that is compatible with all of the dependencies
  ```sh
  conda create -n femo python=3.9.10
  ```
  (Python 3.9.7 also works if Python 3.9.10 is unavailable in your conda)
- Activate the conda enviroment 
  ```sh
  conda activate femo
  ```
- Install the latest FEniCSx by `conda-forge`
  ```sh
  conda install -c conda-forge fenics-dolfinx=0.5.1
  ```
- Install `CSDL_alpha` and `ModOpt` by
  ```sh
  pip install git+https://github.com/LSDOlab/CSDL_alpha.git
  pip install git+https://github.com/LSDOlab/modopt.git
  ```
- Install `femo_alpha` by 
  ```sh
  pip install git+https://github.com/LSDOlab/femo_alpha.git
  ```

## Installation instructions for developers
To install `femo_alpha` as a developer, first clone the repository and install using pip.
On the terminal or command line, run
```sh
git clone https://github.com/LSDOlab/femo_alpha.git
pip install -e ./femo_alpha
```

# License
This project is licensed under the terms of the **GNU Lesser General Public License v3.0**.
