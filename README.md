# KnotGym

[[Website]](https://lil-lab.github.io/knotgym/) [[Paper]](https://arxiv.org/pdf/2505.18028)

A minimalistic environment for spatial reasoning.

## Installation instructions

Step 0:

```sh
uv venv && source .venv/bin/activate
```

or your favorite environment manager

Step 1:

<!-- ```sh
pip install knotgym
```

Alternatively -->

```sh
git clone https://github.com/lil-lab/knotgym.git
uv pip install -e knotgym
```

Step 2: Install `pyknotid` dependency with cython extension

```sh
git clone https://github.com/SPOCKnots/pyknotid.git
cd pyknotid && uv pip install -e . && python setup.py build_ext --inplace && cd ..
```

To verify

```sh
python -c "from pyknotid.spacecurves import chelpers"  # no error/warning
```

We rely on these `chelpers` to compute Gauss code. The speedup is quite significant, if you plan on training with them.

Step 3: Open MuJoCo visualizer

```sh
# MacOS
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
