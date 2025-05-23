"""
python scripts/mjc_post_collect.py --proj --gc --render
python scripts/mjc_post_collect.py --summary
"""

import logging
import os
from collections import Counter
from pprint import pprint

import mediapy as media
import numpy as np
from absl import app, flags
from tqdm import tqdm

import knotgym.specs
import knotgym.utils as knot_utils
from knotgym.specs import (
  CONFIG_BASE_DIR,
  RE_DIR,
  KnotState,
)
from knotgym.utils import colorful

flags.DEFINE_bool("gc", False, "add gauss code")
flags.DEFINE_bool("proj", False, "add projection")
flags.DEFINE_bool("render", False, "add render.npy and render.png")
flags.DEFINE_bool("summary", False, "print summary of configurations")


FLAGS = flags.FLAGS


def safe_write(file_path, content) -> None:
  if os.path.exists(file_path):
    with open(file_path, "r") as f:
      old_content = f.read()
    if old_content == content:
      logging.warning(f"{file_path} exists with the same content, skipping")
      return
    logging.warning(f"different old content: {old_content}")
    logging.warning(f"replacing with new content: {content}")
  with open(file_path, "w") as f:
    f.write(content)


def fn(dir: str, mj_model=None, mj_data=None, renderer=None):
  arr = np.loadtxt(dir + "xpos.txt", converters=float)

  if FLAGS.gc:
    gc = knot_utils.gauss_code(arr)
    safe_write(dir + "gc.txt", str(gc))
  if FLAGS.proj:
    knot = knot_utils._create_knot(arr)
    fig, ax = knot.plot_projection(mark_start=True, show=False)
    fig.savefig(dir + "proj.png")
  if FLAGS.render:
    import mujoco

    qpos = np.loadtxt(dir + "qpos.txt")
    mj_data.qpos[:] = np.copy(qpos)
    mj_data.qvel[:] = np.zeros_like(mj_data.qvel)
    if mj_model.na == 0:
      mj_data.act[:] = None
    mujoco.mj_forward(mj_model, mj_data)

    renderer.update_scene(mj_data, camera="track")
    pixels = renderer.render()
    media.write_image(dir + "render.png", pixels)
    np.save(dir + "render.npy", pixels)


def main(_):
  if FLAGS.render:
    import mujoco

    mj_model = mujoco.MjModel.from_xml_path(
      CONFIG_BASE_DIR.parent / "unknot7_float.xml"
    )
    mj_model.geom_rgba[:] = np.array(
      [
        colorful(i / len(mj_model.geom_rgba))
        for i in range(len(mj_model.geom_rgba))
      ]
    )
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, 512, 512)
  else:
    mj_model = None
    mj_data = None
    renderer = None

  all_dirs = [
    d
    for d in knotgym.specs.CONFIG_BASE_DIR.glob("*")
    if d.is_dir() and RE_DIR.match(d.name)
  ]
  for dir in tqdm(all_dirs, desc="Processing directories"):
    fn(str(dir) + "/", mj_model=mj_model, mj_data=mj_data, renderer=renderer)

  if FLAGS.summary:
    states = [KnotState.load(str(dir.name)) for dir in all_dirs]
    gc_freq = Counter([s.n_crossings for s in states])
    print("(#x, counts)")
    pprint(gc_freq.most_common())
    for c in sorted(gc_freq.keys()):
      _l = [s for s in states if s.n_crossings == c]
      _l = sorted(_l, key=lambda s: s.gc)
      _s = "\n\t".join([str(s) for s in _l])

      print(
        f"n_crossings: {c} | count: {len(_l)} | split {Counter([s.split for s in _l])}\n\t{_s}\n"
      )


if __name__ == "__main__":
  app.run(main)
