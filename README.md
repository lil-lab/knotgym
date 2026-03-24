# KnotGym

[[Website]](https://lil-lab.github.io/knotgym/) [[Paper]](https://arxiv.org/pdf/2505.18028)

A minimalistic environment for spatial reasoning.

## Installation instructions

Step 1:

```sh
git clone --recursive https://github.com/lil-lab/knotgym.git
```

```sh
cd knotgym/
uv venv
uv pip install -e .
```

Step 2: Install `pyknotid` dependency with cython extension

```sh
cd external/pyknotid
uv pip install -e .
uv run python setup.py build_ext --inplace
cd ../..
```

To verify

```sh
uv run python -c "from pyknotid.spacecurves import chelpers"  # no error/warning
```

We rely on these `chelpers` to compute Gauss code. The speedup is quite significant, if you plan on training with them.

Step 3: Open MuJoCo visualizer

```sh
# Locating mjpython could be tricky on MacOS
cd knotgym
MUJOCO_GL=glfw mjpython script/mjc_interact.py
```

## Baselines

See baselines/README.md

## Cite

```bib
@misc{chen2025knotsimpleminimalisticenvironment,
      title={Knot So Simple: A Minimalistic Environment for Spatial Reasoning}, 
      author={Zizhao Chen and Yoav Artzi},
      year={2025},
      eprint={2505.18028},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.18028}, 
}
```
