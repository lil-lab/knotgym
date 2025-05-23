import json
import logging
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import mujoco
import numpy as np
from gymnasium import Wrapper, spaces, utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.wrappers import RescaleAction

import knotgym.specs as specs
from knotgym.mjcf import load_xml_from_asset
from knotgym.utils import colorful, eq_knot, number_of_crossings

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class KnotEnvBase(MujocoEnv, utils.EzPickle):
  r"""The unknot environment for MuJoCo.
  The knot is a passive object modelled as a close chain of beads.
  """

  metadata = {
    "render_modes": ["human", "rgb_array", "depth_array"],
  }

  def __init__(
    self,
    task: str,
    split: str = "tr",
    xml_file: str = "unknot7_float",  # path to the final xml file
    frame_skip: int = 24,  # model.opt.timestep = 0.01 sec -> 0.1 sec/frame
    a_coord_max: float = 2.0,  # >0, bounding the action space for bead coord
    a_frc_max: float = 0.2,  # >0, bounding the action space
    reset_noise_scale: float = 0.015,  # >0, the scale of noise for reset
    reset_frame_skip: int = 0,  # >0, the frame skip for reset
    output_pixels: bool = True,  # whether to output pixels
    render_both: bool = True,  # whether to render both current and goal
    task_max_n_states: Optional[int] = None,
    task_max_n_crossings: Optional[int] = None,
    task_subset_seed: int = -1,  # negative indicates in file order
    **kwargs,
  ):
    assert reset_frame_skip >= 0
    xml_file = load_xml_from_asset(xml_file)
    self.a_coord_max = a_coord_max
    self.a_frc_max = a_frc_max
    self.reset_noise_scale = reset_noise_scale
    self.reset_frame_skip = reset_frame_skip
    self.output_pixels = output_pixels
    self.render_both = render_both
    self._bad_qacc_warned = False
    default_camera_config = {"distance": 2.0}
    kwargs["camera_name"] = "track"
    kwargs["render_mode"] = "rgb_array"

    utils.EzPickle.__init__(
      self,
      task,
      xml_file,
      frame_skip,
      a_coord_max,
      a_frc_max,
      reset_noise_scale,
      output_pixels,
      render_both,
      default_camera_config,
      **kwargs,
    )

    MujocoEnv.__init__(
      self,
      xml_file,
      frame_skip=frame_skip,
      observation_space=None,  # needs to be defined after
      default_camera_config=default_camera_config,
      **kwargs,
    )

    logger.debug(f"frame_skip: {self.frame_skip}")
    logger.debug(f"timestep: {self.model.opt.timestep}")
    logger.debug(f"dt: {self.dt}")

    self.metadata = {
      "render_modes": [
        "human",
        "rgb_array",
        "depth_array",
      ],
      "render_fps": int(np.round(1.0 / self.dt)),
    }
    self.task = specs.parse(task)
    self.spec_factory = specs.init(
      self.task,
      split=split,
      height=self.height,
      width=self.width,
      max_n_states=task_max_n_states,
      max_n_crossings=task_max_n_crossings,
      subset_seed=task_subset_seed,
    )
    self.spec_factory.check(nq=self.model.nq, nx=self.model.nbody - 1)
    self.task_spec: specs.KnotTaskSpec = None  # initialized in reset_model

    pairs = [
      (i, mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i))
      for i in range(self.data.xpos.shape[0])
    ]
    logger.debug("body index to name")
    logger.debug(pairs)

    self.num_beads = self.model.nbody - 1

    colors_nb_4 = np.array(
      [colorful(i / self.num_beads) for i in range(self.num_beads)]
    )
    self.model.geom_rgba[:] = colors_nb_4

    if output_pixels:
      multiplier = 2 if render_both else 1
      self.observation_space = spaces.Box(
        low=0,
        high=255,
        shape=(self.height, self.width * multiplier, 3),
        dtype=np.uint8,
      )
      self.observation_structure = {
        "pixel": (self.height, self.width * multiplier, 3),
      }
    else:
      obs_size = (
        self.num_beads * 3 + self.num_beads * 3 + self.action_space.shape[0]
      )
      self.observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(obs_size,),
        dtype=np.float32,
      )
      self.observation_structure = {
        "xpos": self.num_beads * 3,
        "xposg": self.num_beads * 3,
        "ctrl": self.action_space.shape[0],
      }

    # Keep updated
    self.info_structure = {
      "act/body_index": (),
      "act/body_xfrc": (3,),
      "act/body_xfrc_mag": (),
      "act/body_xfrc_x": (),
      "act/body_xfrc_y": (),
      "act/body_xfrc_z": (),
      "obs/xpos": (self.num_beads, 3),
      "obs/ctrl": self.action_space.shape,
      "is_success": (),
      # episodic stats: prefix with ep_<agg>
      "ep_max/bad_qacc_num": (),
    }

  def _set_action_space(self):
    """overwrite mojoco_env._set_action_space.
    we control with xfrc_applied instead of actuators
    """
    ran = [self.a_coord_max] * 3 + [self.a_frc_max] * 3
    ran = np.array(ran, dtype=np.float64)
    action_space = spaces.Box(
      low=-ran,
      high=ran,
      shape=(len(ran),),
      dtype=np.float64,
    )
    self.action_space = action_space
    return action_space

  def _ctrl_to_xfrc(self, ctrl) -> Tuple[int, np.ndarray]:
    """convert ctrl to xfrc_applied"""
    frc_coord, bead_xfrc = ctrl[:3], ctrl[3:]
    # find closest bead
    frc_dist = np.linalg.norm(self.data.xpos[1:] - frc_coord, axis=-1)
    bead_index = np.argmin(frc_dist)
    # offset by the first body which is the world
    body_index = bead_index + 1
    return body_index, bead_xfrc

  def do_simulation(self, ctrl):
    """overwrite MujocoEnv.do_simulation, MujocoEnv._step_mujoco_simulation
    to apply xfrc_applied instead of actuator via data.ctrl
    """
    if not self.action_space.contains(ctrl):
      raise ValueError("Action is not in action space")

    body_index, body_xfrc = self._ctrl_to_xfrc(ctrl)

    # apply force to the body after resetting
    self.data.xfrc_applied[:, :] = 0.0
    self.data.xfrc_applied[body_index, :3] = body_xfrc

    mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

    _bad_qacc_num = self.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number
    if _bad_qacc_num > 0 and not self._bad_qacc_warned:
      logger.warning(
        f"bad qacc {_bad_qacc_num} at time {self.data.time:.4f} from {self.task_spec.dir0}"
      )
      self._bad_qacc_warned = True
    sim_info = {"ep_max/bad_qacc_num": _bad_qacc_num}
    return self._get_act_info(body_index, body_xfrc) | sim_info

  def _get_act_info(self, body_index: int, body_xfrc: np.ndarray):
    return {
      "act/body_index": body_index,  # int
      "act/body_xfrc": body_xfrc,  # (3,)
      "act/body_xfrc_mag": np.linalg.norm(body_xfrc),  # float
      "act/body_xfrc_x": body_xfrc[0],  # float
      "act/body_xfrc_y": body_xfrc[1],  # float
      "act/body_xfrc_z": body_xfrc[2],  # float
    }

  def step(self, action):
    act_info = self.do_simulation(action)
    obs, obs_info = self._get_obs(action)

    bad_qacc = act_info["ep_max/bad_qacc_num"] > 0
    done = bad_qacc  # will trigger reset
    terminated = bad_qacc
    reward = -10.0 if bad_qacc else 0.0  # annealing if bad qacc occurs

    rew_info = {}
    info = act_info | obs_info | rew_info | {"is_success": False}
    if self.render_mode == "human":
      self.render()
    return obs, reward, done, terminated, info

  def _get_obs(self, ctrl: np.ndarray):
    info = {
      "obs/xpos": self.data.xpos[1:],  # exclude the ground
      "obs/ctrl": ctrl,
    }
    if self.output_pixels:
      obs = self.render()
      return obs, info
    xpos = self.data.xpos[1:].flatten().astype(np.float32)
    xposg = self.task_spec.xposg.flatten().astype(np.float32)
    ctrl = ctrl.astype(np.float32)
    obs = np.concatenate([xpos, xposg, ctrl])
    return obs, info

  def reset_model(self):
    spec = self.spec_factory.generate_task_spec(self.np_random)
    spec = spec.jiggle(self.np_random, noise_scale=self.reset_noise_scale)
    # setting the spec
    self.task_spec = spec
    qvel = self.init_qvel
    self.set_state(spec.qpos0, qvel)

    # make sure this resets every episode
    assert self.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number == 0
    self._bad_qacc_warned = False

    dummy_ctrl = np.zeros(self.action_space.shape, dtype=np.float64)
    for _ in range(self.reset_frame_skip):
      self.step(dummy_ctrl)
    obs, _ = self._get_obs(dummy_ctrl)
    return obs

  def _get_reset_info(self):
    dummy_ctrl = np.zeros(self.action_space.shape, dtype=np.float64)
    _, obs_info = self._get_obs(dummy_ctrl)
    body_index, body_xfrc = self._ctrl_to_xfrc(dummy_ctrl)
    act_info = self._get_act_info(body_index, body_xfrc)
    return act_info | obs_info | {"is_success": False, "ep_max/bad_qacc_num": 0}

  def render(self):
    orig_rendered = super().render()
    if not self.render_both:
      return orig_rendered
    goal_rendered = self.task_spec.obsg
    # left: original, right: goal
    return np.hstack((orig_rendered, goal_rendered))  # (H, W, 3) -> (H, 2W, 3)


## Wrappers


class UnknotRewardWrapper(Wrapper):
  def __init__(
    self,
    env,
    r_scale_zero_cross: float = 1.0,  # >0 for the final reward based on zero crossings
    r_scale_dt_cross: float = -0.0,  # <0 for punishing inc crossings / rewarding dec crossings
  ):
    super().__init__(env)
    self.r_scale_zero_cross = r_scale_zero_cross
    self.r_scale_dt_cross = r_scale_dt_cross
    self._num_cross = None  # stateful
    add_info_structure = {"reward/num_cross": (), "reward/dt_num_cross": ()}
    self.set_wrapper_attr(
      "info_structure",
      env.get_wrapper_attr("info_structure") | add_info_structure,
    )

  def step(self, action):
    obs, _, done, truncated, info = self.env.step(action)
    reward2, info2 = self._get_reward(info)
    return obs, reward2, done, truncated, info2

  def _get_reward(self, info):
    xpos = info["obs/xpos"]
    num_cross = number_of_crossings(xpos)
    rew_zero_cross = 1.0 if num_cross == 0 else 0.0
    dt_num_cross = num_cross - self._num_cross
    rew = (
      dt_num_cross * self.r_scale_dt_cross
      + rew_zero_cross * self.r_scale_zero_cross
    )
    info["is_success"] = rew_zero_cross
    self._num_cross = num_cross
    return rew, info | {
      "reward/num_cross": num_cross,
      "reward/dt_num_cross": dt_num_cross,
    }

  def reset(self, **kwargs):
    obs, info = self.env.reset(**kwargs)
    self._num_cross = number_of_crossings(info["obs/xpos"])
    info["is_success"] = 1.0 if self._num_cross == 0 else 0.0
    return obs, info | {
      "reward/num_cross": self._num_cross,
      "reward/dt_num_cross": 0,
    }


class TieRewardWrapper(Wrapper):
  """task 2: tie a knot such that the gaussian code is the same as the specified one"""

  def __init__(
    self,
    env,
    r_scale_gc: float = 1.0,
    r_scale_coord: float = 0.0,
    r_gc_allow_flipped_or_mirrored: bool = False,
    r_gc_allow_chiral_all: bool = False,
  ):
    super().__init__(env)
    self.r_scale_gc = r_scale_gc
    self.r_scale_coord = r_scale_coord
    self.r_gc_allow_flipped_or_mirrored = r_gc_allow_flipped_or_mirrored
    self.r_gc_allow_chiral_all = r_gc_allow_chiral_all
    if r_gc_allow_chiral_all and not r_gc_allow_flipped_or_mirrored:
      logger.warning(
        f"{r_gc_allow_chiral_all=} will override {r_gc_allow_flipped_or_mirrored=}"
      )
    add_info_structure = {
      "reward/gc": (),
      "reward/coord": (),
    }
    self.set_wrapper_attr(
      "info_structure",
      env.get_wrapper_attr("info_structure") | add_info_structure,
    )

  @property
  def _task_spec(self) -> specs.KnotTaskSpec:
    return self.unwrapped.task_spec

  def step(self, action):
    obs, _, done, truncated, info = self.env.step(action)
    reward2, info2 = self._get_reward(info)
    return obs, reward2, done, truncated, info2

  def _get_reward(self, info):
    xpos = info["obs/xpos"]
    xposg = self._task_spec.xposg
    is_eq_gc = eq_knot(
      "gc",
      xpos,
      xposg,
      allow_flipped_or_mirrored=self.r_gc_allow_flipped_or_mirrored,
      allow_chiral_all=self.r_gc_allow_chiral_all,
    )
    gc = 1.0 if is_eq_gc else 0.0
    coord_dist = np.linalg.norm(xpos - xposg)
    rew = gc * self.r_scale_gc + coord_dist * self.r_scale_coord
    info["is_success"] = is_eq_gc
    return rew, info | {"reward/gc": gc, "reward/coord": coord_dist}

  def reset(self, **kwargs):
    obs, info = self.env.reset(**kwargs)
    _, info2 = self._get_reward(info)
    return obs, info2


class TimeLimit(Wrapper):
  def __init__(self, env, duration: int, punish: float):
    assert punish <= 0
    assert duration > 0
    logger.debug("TimeLimit")
    super().__init__(env)
    self.duration = duration
    self.punish = punish

    # used by stable_baselines3.common.OnPolicyAlgorithm.collect_rollouts
    self.set_wrapper_attr(
      "info_structure",
      env.get_wrapper_attr("info_structure")
      | {
        "rew/timeout_punish": (),
        "TimeLimit.truncated": (),
      },
    )
    self.set_wrapper_attr("max_episode_steps", duration)
    self._len = None

  def step(self, action):
    obs, rew, done, truncated, info = self.env.step(action)
    self._len += 1
    _punish = 0.0
    if self._len >= self.duration:
      truncated = True
      if not done:  # only punish if not done yet
        rew += self.punish
        _punish = self.punish
    info["rew/timeout_punish"] = _punish
    info["TimeLimit.truncated"] = truncated
    return obs, rew, done, truncated, info

  def reset(self, **kwargs):
    self._len = 0
    obs, info = super().reset(**kwargs)
    return obs, info | {
      "rew/timeout_punish": 0.0,
      "TimeLimit.truncated": False,
    }


class TerminalObservation(Wrapper):
  # used by stable_baselines3.common.OnPolicyAlgorithm.collect_rollouts
  def __init__(self, env):
    super().__init__(env)
    self.set_wrapper_attr(
      "info_structure",
      env.get_wrapper_attr("info_structure")
      | {
        "terminal_observation": self.observation_space.shape,
      },
    )

  def step(self, action):
    obs, rew, done, truncated, info = self.env.step(action)
    term_obs = None
    if done or truncated:
      term_obs = obs
    info["terminal_observation"] = term_obs
    return obs, rew, done, truncated, info


def timestamp() -> str:
  return datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")


class EpisodicSaver(Wrapper):
  """log everything per step for post-hoc replay"""

  def __init__(
    self,
    env,
    logdir: str,
    logfreq: int = 1,  # every n episode
    record_info_keys=(
      "obs/xpos",
      "obs/ctrl",
      "ep_max/bad_qacc_num",
      "act/body_xfrc_mag",
      "act/body_index",
    ),
  ):
    self._logdir = Path(logdir)
    self._logdir.mkdir(parents=True, exist_ok=True)
    self._buffer: Dict[str, List] = dict()
    self._info_keys = record_info_keys
    super().__init__(env)
    assert all(
      k in self.env.get_wrapper_attr("info_structure").keys()
      for k in self._info_keys
    )
    self._logfreq = logfreq
    self._ep_idx = 0

  @property
  def _enabled(self) -> bool:
    return self._ep_idx % self._logfreq == 0

  def step(self, action):
    obs, rew, done, truncated, info = self.env.step(action)

    for k in self._info_keys:
      v = info[k]
      if isinstance(v, np.ndarray):
        v = v.copy()
      self._buffer[k].append(v)
    self._buffer["action"].append(action)
    self._buffer["reward"].append(rew)
    self._buffer["done"].append(done)
    self._buffer["truncated"].append(truncated)
    if (done or truncated) and self._enabled:
      self._write_buffer()
    return obs, rew, done, truncated, info

  def _write_buffer(self):
    # stepwise
    if self._buffer == dict() or self._buffer["done"] == []:
      logger.warning(f"empty buffer. no data to save for {self._logdir_curr}")
      return
    for k, v in self._buffer.items():
      self._buffer[k] = np.array(v)
    step_path = self._logdir_curr / "stepwise.npz"
    np.savez(step_path, **self._buffer)
    logger.debug(f"saved to {step_path}")

    # episodic
    spec = self.unwrapped.task_spec
    out = {
      "spec": {
        "task": str(spec.task),
        "dir0": spec.dir0,
        "gc0": str(spec.gc0),
        "dirg": spec.dirg,
        "gcg": str(spec.gcg),
        "n_crossings0": spec.n_crossings0,
        "n_crossingsg": spec.n_crossingsg,
      },
      "length": len(self._buffer["done"]),
      "score": np.sum(self._buffer["reward"]).item(),
      "is_success": np.any(self._buffer["done"]).item(),
      "_ep_idx": self._ep_idx,
    }
    for k, v in self._buffer.items():
      if v.ndim == 0:
        out[k] = v.item()
      if v.ndim == 1:
        out[k] = np.mean(v).item()
      else:
        pass  # ignore action or other high dimensional data
    ep_path = self._logdir_curr / "episodic.json"
    with open(ep_path, "w") as f:
      json.dump(out, f, indent=2, sort_keys=True)
    logger.debug(f"saved to {ep_path}")

  def _clear_buffer(self):
    default_keys = ("action", "reward", "done", "truncated")
    for k in default_keys + self._info_keys:
      self._buffer[k] = deque(maxlen=1000)

  def reset(self, **kwargs):
    self._ep_idx += 1
    ret = self.env.reset(**kwargs)
    if self._enabled:
      self._logdir_curr = self._logdir / timestamp()
      self._logdir_curr.mkdir()
    self._clear_buffer()
    return ret


class KnotEnv(Wrapper):
  metadata = KnotEnvBase.metadata

  def __init__(
    self,
    *args,
    normalize_action: bool = True,
    r_scale_dt_cross: float = 0.0,
    r_scale_zero_cross: float = 5.0,  # same as the gc scale
    r_scale_gc: float = 5.0,
    r_scale_coord: float = 0.0,
    r_gc_allow_flipped_or_mirrored: bool = True,
    r_gc_allow_chiral_all: bool = True,
    duration: int = 50,
    r_scale_timeout_punish: float = -5.0,
    logdir: Optional[Path] = None,
    logfreq: int = 1,
    **kwargs,
  ):
    env = KnotEnvBase(*args, **kwargs)
    task = env.task

    if normalize_action:
      env = RescaleAction(env, min_action=-1.0, max_action=1.0)

    if task == specs.KnotTask.UNKNOT:
      env = UnknotRewardWrapper(
        env,
        r_scale_dt_cross=r_scale_dt_cross,
        r_scale_zero_cross=r_scale_zero_cross,
      )
    elif task == specs.KnotTask.TIE or task == specs.KnotTask.EQ1:
      env = TieRewardWrapper(
        env,
        r_scale_gc=r_scale_gc,
        r_scale_coord=r_scale_coord,
        r_gc_allow_flipped_or_mirrored=r_gc_allow_flipped_or_mirrored,
        r_gc_allow_chiral_all=r_gc_allow_chiral_all,
      )
    else:
      raise NotImplementedError(f"{task=}")

    if duration > 0:
      assert r_scale_timeout_punish <= 0
      env = TimeLimit(env, duration=duration, punish=r_scale_timeout_punish)

    env = TerminalObservation(env)

    # passive wrappers outside active wrappers
    # if other termination changing wrappers are used, this will not work properly
    if logdir:
      env = EpisodicSaver(env, logdir, logfreq=logfreq)

    super().__init__(env)
    self.metadata = self.unwrapped.metadata


if __name__ == "__main__":
  env = KnotEnv(
    task="tie",
    xml_file="unknot7_float",
    height=128,
    width=128,
    logdir="tmp",
    duration=20,
  )

  obs, _ = env.reset()

  frames = [obs]
  done = False
  truncated = False
  while not (done or truncated):
    obs, rew, done, truncated, info = env.step(env.action_space.sample())
    frames.append(obs)
  env.close()

  import imageio

  array = np.array(frames)
  array = (array * 255).astype(np.uint8)
  imageio.mimsave("debug_render.gif", array, fps=10)
  logger.info("saved gif to debug_render.gif")
