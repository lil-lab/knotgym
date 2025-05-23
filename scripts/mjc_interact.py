"""
# on macos
MUJOCO_GL=glfw mjpython scripts/mjc_interact.py --mjcf ../src/knotgym/assets/unknot7_float.xml --random
"""

import os

use_gpu = False
if use_gpu:
  os.environ["MUJOCO_GL"] = "egl"

import time
import xml.etree.ElementTree as ET

import mediapy as media
import mujoco
import numpy as np
from absl import app, flags, logging
from mujoco.viewer import launch_passive
from datetime import datetime

from knotgym.mjcf import initialize_knot_coords
import knotgym.utils as knot_utils

flags.DEFINE_string(
  "mjcf", "../src/knotgym/knotgym/assets/unknot7_float.xml", "mjcf path"
)
flags.DEFINE_string("mjcf_init", "../results/data.txt", "initial vertices")
flags.DEFINE_enum(
  "mjcf_build", "none", ["none", "simple", "full"], "build mjcf"
)
flags.DEFINE_bool("record", False, "record video")
flags.DEFINE_bool("expert", False, "assert expert action")
flags.DEFINE_bool("random", False, "assert random action")
flags.DEFINE_bool("bare", False, "remove callbacks and viewer etc")
flags.DEFINE_string(
  "output_dir", "../src/knotgym/assets/configurations", "save dir"
)

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


def configure_display():
  """misc display configurations"""
  if use_gpu:
    NVIDIA_ICD_CONFIG_PATH = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    if not os.path.exists(NVIDIA_ICD_CONFIG_PATH):
      with open(NVIDIA_ICD_CONFIG_PATH, "w") as f:
        f.write("""{
            "file_format_version" : "1.0.0",
            "ICD" : {
                "library_path" : "libEGL_nvidia.so.0"
            }
        }
        """)

  np.set_printoptions(precision=2, suppress=True)
  logging.get_absl_handler().setFormatter(None)


class Callback:
  def __init__(self):
    pass

  def on_key(self, *args, **kwargs):
    pass

  def on_step(self, *args, **kwargs):
    pass

  def on_open(self, *args, **kwargs):
    pass

  def on_close(self, *args, **kwargs):
    pass


class CallbackList(Callback):
  def __init__(self):
    self.callbacks = []

  def add(self, cb: Callback):
    self.callbacks.append(cb)

  def on_key(self, *args, **kwargs):
    for cb in self.callbacks:
      cb.on_key(*args, **kwargs)

  def on_step(self, *args, **kwargs):
    for cb in self.callbacks:
      cb.on_step(*args, **kwargs)

  def on_open(self, *args, **kwargs):
    for cb in self.callbacks:
      cb.on_open(*args, **kwargs)

  def on_close(self, *args, **kwargs):
    for cb in self.callbacks:
      cb.on_close(*args, **kwargs)


class PrinterCb(Callback):
  def __init__(self, log_every_n=20):
    self.log_every_n = log_every_n

  def on_step(self, mj_model, mj_data, *args, **kwargs):
    n = self.log_every_n
    logging.log_every_n(logging.INFO, "step: %s", n, round(mj_data.time, 4))

    xfrc_applied = mj_data.xfrc_applied  # 101x6
    non_zero_index = np.where(xfrc_applied != 0)
    logging.log_every_n(
      logging.INFO, "xfrc_applied: %s", n, xfrc_applied[non_zero_index]
    )

    body_0_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, 0)
    logging.log_every_n(logging.INFO, "body at 0: %s", n, body_0_name)
    logging.log_every_n(logging.INFO, "body xpos at 0: %s", n, mj_data.xpos[0])
    logging.log_every_n(
      logging.INFO, "qfrc_applied: %s", n, mj_data.qfrc_applied.shape
    )
    logging.log_every_n(
      logging.INFO, "xfrc_applied: %s", n, mj_data.xfrc_applied.shape
    )
    logging.log_every_n(logging.INFO, "xpos: %s", n, mj_data.xpos.shape)
    logging.log_every_n(
      logging.INFO, "gc: %s", n, str(knot_utils.gauss_code(mj_data.xpos[1:]))
    )
    gc = knot_utils.gauss_code(mj_data.xpos[1:])
    erased_gc = str(gc).replace("a", "").replace("c", "")
    logging.log_every_n(logging.INFO, "erased_gc: %s", n, erased_gc)

  def on_open(self, *args, **kwargs):
    logging.info("open")

  def on_close(self, *args, **kwargs):
    logging.info("close")


class VideoRecorderCb(Callback):
  def __init__(self, fps=60, path="results/video.mp4"):
    self.fps = fps
    self.path = path
    self.frames = None
    self.renderer = None

  def on_open(self, mj_model, *args, **kwargs):
    self.renderer = mujoco.Renderer(mj_model)
    self.frames = []

  def on_step(self, mj_model, mj_data, *args, **kwargs):
    """opt: grab rendered pixels from gui and save as video"""
    if len(self.frames) < self.fps * mj_data.time:
      self.renderer.update_scene(mj_data)
      pixels = self.renderer.render()
      self.frames.append(pixels)

  def on_close(self, *args, **kwargs):
    media.write_video(self.path, self.frames, fps=self.fps)
    logging.info("video saved to %s", self.path)
    self.renderer.close()
    self.frames = None


class RandomInputCb(Callback):
  def __init__(self, log_every_n=20):
    self.log_every_n = log_every_n

  def on_step(self, mj_model, mj_data, *args, **kwargs):
    ctrl = np.random.uniform(-1, 1, (4,))
    scale = np.array([1.0, 0.2, 0.2, 0.2])
    ctrl = ctrl * scale
    number_of_beads = mj_model.nbody - 1
    body_index, body_xfrc = self._ctrl_to_xfrc(
      ctrl, number_of_beads=number_of_beads
    )
    mj_data.xfrc_applied[:, :] = 0.0
    mj_data.xfrc_applied[body_index, :3] = body_xfrc

  def _ctrl_to_xfrc(self, ctrl, number_of_beads):
    bead_index = int((ctrl[0] + 1) * (number_of_beads - 1) / 2)
    body_index = bead_index + 1
    bead_xfrc = ctrl[1:]  # 3d force
    return body_index, bead_xfrc


class StepTimerCb(Callback):
  # goal: 10k steps/sec, 0.1 ms/step, 0.0001 s/step
  def __init__(self, log_every_n=20):
    self.tic = 0
    self.log_every_n = log_every_n

  def on_step(self, *args, **kwargs):
    if not self.tic:
      self.tic = time.time()
      return
    toc = time.time()
    dt = toc - self.tic
    self.tic = toc
    logging.log_every_n(
      logging.INFO,
      f"step: {round(dt * 1e3, 2)} ms | num steps/sec: {int(1 / dt)}",
      self.log_every_n,
    )


class ConfigurationSaverCb(Callback):
  def __init__(self, key):
    self.key = key
    self._pending_saving = False
    self._dir = FLAGS.output_dir
    self._n_decimals = 4
    self._renderer = None

  def on_key(self, key, *args, **kwargs):
    if key == ord(self.key):
      self._pending_saving = True

  def _maybe_init_renderer(self, mj_model):
    if self._renderer is None:
      self._renderer = mujoco.Renderer(mj_model, height=512, width=512)

  def on_step(self, mj_model, mj_data, *args, **kwargs):
    self._maybe_init_renderer(mj_model)
    if self._pending_saving:
      time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
      _dir = os.path.join(self._dir, time_str)
      os.mkdir(_dir)

      _fmt = f"%.{self._n_decimals}f"
      np.savetxt(os.path.join(_dir, "qpos.txt"), mj_data.qpos, fmt=_fmt)
      np.savetxt(os.path.join(_dir, "xpos.txt"), mj_data.xpos[1:], fmt=_fmt)

      self._renderer.update_scene(mj_data, camera="track")
      pixels = self._renderer.render()
      media.write_image(os.path.join(_dir, "render.png"), pixels)
      np.save(os.path.join(_dir, "render.npy"), pixels)

      gc = knot_utils.gauss_code(mj_data.xpos[1:])
      safe_write(os.path.join(_dir, "gc.txt"), str(gc))

      split = "train" if np.random.rand() < 0.5 else "val"
      safe_write(os.path.join(_dir, "split.txt"), split)

      logging.info("saved to %s", _dir)
      self._pending_saving = False


class PerturbCb(Callback):
  """perturb qpos with noise by pressing key"""

  def __init__(self, key, noise_scale):
    self.noise_scale = noise_scale
    self._pending_perturb = False
    self.key = key

  def on_key(self, key, *args, **kwargs):
    if key == ord(self.key):
      self._pending_perturb = True

  def on_step(self, mj_model, mj_data, *args, **kwargs):
    if self._pending_perturb:
      n = len(mj_data.qpos)
      l = self.noise_scale
      noise = np.random.uniform(-l, l, n)
      mj_data.qpos += noise
      logging.info("perturb qpos with noise")
      self._pending_perturb = False


def setup_callbacks() -> Callback:
  cb = CallbackList()
  log_every_n = 50
  cb.add(PrinterCb(log_every_n=log_every_n))
  cb.add(StepTimerCb(log_every_n=log_every_n))
  cb.add(ConfigurationSaverCb(key=" "))
  cb.add(PerturbCb(key="/", noise_scale=0.01))
  if FLAGS.record:
    cb.add(VideoRecorderCb())
  if FLAGS.random:
    cb.add(RandomInputCb(log_every_n=log_every_n))
  return cb


def build_mjcf():
  template_path = FLAGS.mjcf
  data_path = FLAGS.mjcf_init
  build_strategy = FLAGS.mjcf_build
  root = ET.parse(template_path).getroot()
  if build_strategy == "none":
    logging.info("no build strategy, use original mjcf")
    pass
  elif build_strategy == "simple":
    logging.info(f"building mjcf from {template_path}")
    logging.info(f"initializing vertices with {data_path}")
    with open(data_path, "r") as f:
      vertices = f.read().replace("\n", " ")
    composites = root.findall(".//composite[@type='cable']")
    assert len(composites) == 1, "Only one cable object is supported"
    composite = composites[0]
    composite.set("vertex", vertices)
    composite.attrib.pop("count")
    composite.attrib.pop("curve")
    composite.attrib.pop("size")
    composite.attrib.pop("offset")
  elif build_strategy == "full":
    xml_filled = initialize_knot_coords(
      template_path,
      data_path,
      num_beads=100,
      num_subcables=8,
    )
    root = ET.parse(xml_filled).getroot()
  else:
    raise ValueError("unknown build strategy")
  return ET.tostring(root)


def colorful(x: float) -> np.ndarray:
  """x: float in [0, 1], returns an RGB color in (4,)"""
  if x == 0.0:
    return np.array([1.0, 1.0, 1.0, 1.0])  # white for the start
  r = np.sin(2 * np.pi * (x + 0.0)) * 0.5 + 0.5
  g = np.sin(2 * np.pi * (x + 0.33)) * 0.5 + 0.5
  b = np.sin(2 * np.pi * (x + 0.67)) * 0.5 + 0.5
  return np.array([r, g, b, 1.0])


def main(_):
  configure_display()
  mjcf = build_mjcf()
  mj_model = mujoco.MjModel.from_xml_string(mjcf)
  # set mj_model.geom_rgba  (101x4)
  mj_model.geom_rgba[:] = np.array(
    [colorful(i / 100) for i in range(len(mj_model.geom_rgba))]
  )
  mj_data = mujoco.MjData(mj_model)  # initially xpos is empty. qpos is m.qpos0
  cb = setup_callbacks()
  kws = dict(show_right_ui=False, key_callback=cb.on_key)

  if FLAGS.bare:
    for _ in range(1000):
      mujoco.mj_step(mj_model, mj_data, nstep=10)
    return

  assert not FLAGS.bare
  with launch_passive(mj_model, mj_data, **kws) as viewer:
    cb.on_open(mj_model)
    while viewer.is_running():
      mujoco.mj_step(mj_model, mj_data, nstep=10)
      viewer.sync()
      cb.on_step(mj_model, mj_data)
  cb.on_close()


if __name__ == "__main__":
  app.run(main)
