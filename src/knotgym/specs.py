import dataclasses
import enum
import logging
import re
from collections import defaultdict
from functools import cache
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from etils import epath

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


CONFIG_BASE_DIR = epath.resource_path("knotgym") / "assets" / "configurations"
# glob that matches "2025-01-01-00-00-00-anytext"
RE_DIR = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}(-.*)?$")


class KnotTask(enum.Enum):
  UNKNOT = enum.auto()  # convert known loop-eq to loop
  TIE = enum.auto()  # convert a loop to a non-trivial loop-eq
  EQ1 = enum.auto()  # assess if two knots are equivalent


def parse(task_str: str) -> KnotTask:
  if task_str == "unknot":
    return KnotTask.UNKNOT
  if task_str == "tie":
    return KnotTask.TIE
  if task_str == "eq1":
    return KnotTask.EQ1
  raise ValueError(f"unknown task: {task_str}")


class KnotTaskSplit(enum.Enum):
  TRAIN = "tr"  # n crossings
  EVAL_A = "ea"  # n crossings, diff configs


@dataclasses.dataclass(frozen=True)
class KnotState:
  xpos: np.ndarray
  qpos: np.ndarray
  obs: np.ndarray
  gc: Tuple[str, ...]
  dir: str
  split: str

  @staticmethod
  @cache
  def load(dir: str) -> "KnotState":
    full_dir = CONFIG_BASE_DIR / dir
    qpos = np.loadtxt(full_dir / "qpos.txt")
    xpos = np.loadtxt(full_dir / "xpos.txt")
    render = np.load(full_dir / "render.npy")
    with open(full_dir / "gc.txt") as f:
      gc = f.read().strip()
    gc = tuple(gc.split(","))
    with open(full_dir / "split.txt") as f:
      split = f.read().strip()
      assert split in ("train", "val")
    return KnotState(
      xpos=xpos, qpos=qpos, obs=render, gc=gc, dir=dir, split=split
    )

  @property
  def n_crossings(self) -> int:
    return len(self.gc) // 2

  def __str__(self):
    return f"{self.dir} ({self.split}) {self.gc} "


@dataclasses.dataclass
class KnotTaskSpec:
  task: KnotTask
  split: KnotTaskSplit
  state0: KnotState
  stateg: KnotState
  xpos0: np.ndarray
  xposg: np.ndarray
  qpos0: np.ndarray
  qposg: np.ndarray
  obs0: np.ndarray
  obsg: np.ndarray
  gc0: Tuple[str]
  gcg: Tuple[str]
  dir0: str
  dirg: str

  def jiggle(self, rng, noise_scale: float = 0.01):
    s = noise_scale
    # make sure the noise does not accumulate every time we call jiggle
    self.qpos0 = self.state0.qpos + rng.uniform(-s, s, size=self.qpos0.shape)
    self.qposg = self.stateg.qpos + rng.uniform(-s, s, size=self.qposg.shape)
    return self

  @classmethod
  def load(
    cls,
    task: KnotTask,
    split: KnotTaskSplit,
    state0: KnotState,
    stateg: KnotState,
  ):
    return KnotTaskSpec(
      task=task,
      split=split,
      state0=state0,
      stateg=stateg,
      xpos0=state0.xpos,
      xposg=stateg.xpos,
      qpos0=state0.qpos,
      qposg=stateg.qpos,
      obs0=state0.obs,
      obsg=stateg.obs,
      gc0=state0.gc,
      gcg=stateg.gc,
      dir0=state0.dir,
      dirg=stateg.dir,
    )

  @property
  def max_n_crossings(self) -> int:
    return max(self.n_crossings0, self.n_crossingsg)

  @property
  def n_crossings0(self) -> int:
    return self.state0.n_crossings

  @property
  def n_crossingsg(self) -> int:
    return self.stateg.n_crossings


class KnotTaskSpecFactory:
  def __init__(
    self,
    task: KnotTask,
    split: str,
    height: int = 128,
    width: int = 128,
    # filtering options
    max_n_crossings=None,
    max_n_states=None,
    subset_seed=-1,
  ):
    # all of these needs to be updated once
    self.task = task
    self.split = split
    self.height = height
    self.width = width
    all_dirs = [d for d in CONFIG_BASE_DIR.glob("*") if RE_DIR.match(d.name)]
    self.all_states = [KnotState.load(d.name) for d in all_dirs]

    l0, lg = self._get_initial_goal_states(self.all_states, max_n_crossings)

    # post filtering
    max_n_states = max_n_states or float("inf")
    if max_n_states < float("inf"):
      logger.debug(f"subset_seed: {subset_seed}")
      logger.debug(f"max_n_states: {max_n_states}")
      if subset_seed == -1:
        l0 = l0[: int(max_n_states)]
        lg = lg[: int(max_n_states)]
      else:
        rng = np.random.default_rng(subset_seed)
        size = min(len(l0), len(lg), int(max_n_states))
        idx0 = rng.choice(len(l0), size=size, replace=False)
        idx0 = sorted(idx0)
        idxg = rng.choice(len(lg), size=size, replace=False)
        idxg = sorted(idxg)
        l0 = [l0[i] for i in idx0]
        lg = [lg[i] for i in idxg]
        logger.debug(f"idx0: {idx0}")
        logger.debug(f"idxg: {idxg}")

    self.l0 = l0  # initial states
    self.lg = lg  # goal states

    logger.debug(
      f"task: {self.task}, split: {self.split}, {len(l0)} initial states, {len(lg)} goal states"
    )
    logger.debug(f"l0: {[str(s) for s in l0]}")
    logger.debug(f"lg: {[str(s) for s in lg]}")

  def _get_initial_goal_states(
    self, all_states: List[KnotState], max_n_crossings: Optional[int] = None
  ) -> Tuple[List[KnotState], List[KnotState]]:
    # group by number of crossings
    all_states_by_crossings = defaultdict(list)
    for s in all_states:
      all_states_by_crossings[s.n_crossings].append(s)

    l_simple: List[KnotState] = all_states_by_crossings[0]
    if max_n_crossings is None:
      max_n_crossings = 1
      logger.warning("max_n_crossings is +inf, defaulting to 1")
    logger.debug(f"targeting nxg: {max_n_crossings}")
    l_complex: List[KnotState] = all_states_by_crossings[max_n_crossings]

    # filter by splits for the list of complex knots
    sp = "train" if self.split == "tr" else "val"
    l_simple = [s for s in l_simple if s.split == sp]
    l_complex = [s for s in l_complex if s.split == sp]

    if self.task == KnotTask.UNKNOT:
      l0 = l_complex
      lg = l_simple
    elif self.task == KnotTask.TIE:
      l0 = l_simple
      lg = l_complex
    elif self.task == KnotTask.EQ1:
      # convert a (n-1) crossings knot to an n crossings knot
      assert max_n_crossings > 1
      l_simple = all_states_by_crossings[max_n_crossings - 1]
      l_simple = [s for s in l_simple if s.split == sp]
      l0 = l_simple
      lg = l_complex
    else:
      raise NotImplementedError(f"{self.task=}")
    return sorted(l0, key=lambda s: s.dir), sorted(lg, key=lambda s: s.dir)

  def check(self, nq, nx):
    assert all([s.xpos.shape == (nx, 3) for s in self.all_states])
    assert all([s.qpos.shape == (nq,) for s in self.all_states])

  def generate_task_spec(self, rng: np.random.Generator) -> KnotTaskSpec:
    l0, lg = self.l0, self.lg
    idx0 = rng.integers(0, len(l0))
    idxg = rng.integers(0, len(lg))
    spec = KnotTaskSpec.load(
      self.task, KnotTaskSplit(self.split), l0[idx0], lg[idxg]
    )
    logger.info(
      f"{spec.task}, (nx0:{spec.state0.n_crossings}){spec.state0}, (nxg:{spec.stateg.n_crossings}){spec.stateg}"
    )
    spec.obs0 = cv2.resize(spec.obs0, (self.width, self.height))
    spec.obsg = cv2.resize(spec.obsg, (self.width, self.height))
    return spec


FACTORIES: Dict[KnotTaskSplit, KnotTaskSpecFactory] = {}


def init(*args, **kwargs):
  split = kwargs["split"]
  if split in FACTORIES:
    return FACTORIES[split]
  factory = KnotTaskSpecFactory(*args, **kwargs)
  FACTORIES[split] = factory
  return factory


__all__ = ["KnotTask", "KnotTaskSpec", "init", "parse"]


if __name__ == "__main__":
  rng = np.random.default_rng()
  factory = init(KnotTask.UNKNOT)
  spec = factory.generate_task_spec(rng)
  logger.info(spec)
