"""
python baselines/random/random_policy.py --seed=0 --num_envs=16 --num_episodes=64 --max_n_crossings=2
"""

import json
import os
import sys
from datetime import datetime
from functools import partial as bind

import numpy as np
from absl import app, flags
from gymnasium.vector import AsyncVectorEnv
import logging
from knotgym.envs import KnotEnv

# parallel
flags.DEFINE_integer("num_envs", 16, "Number of environments to run")
flags.DEFINE_integer("num_episodes", 16, "Number of episodes to run")

# condition
flags.DEFINE_enum("task", "unknot", ["unknot", "tie", "eq1"], "task")
flags.DEFINE_enum("split", "ea", ["ea", "tr"], "split")
flags.DEFINE_integer("max_n_crossings", 1, "Number of crossings in target knot")
flags.DEFINE_integer("episode_length", 50, "Length of each episode")

# reproducibility
flags.DEFINE_integer("seed", 0, "Random seed for reproducibility")

FLAGS = flags.FLAGS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def make_env(
  rank: int,
  task: str,
  split: str,
  max_n_crossings: int,
  episode_length: int,
):
  return KnotEnv(
    task=task,
    split=split,
    task_max_n_crossings=max_n_crossings,
    task_max_n_states=20,
    duration=episode_length,
    output_pixels=False,
    r_gc_allow_flipped_or_mirrored=True,
  )


def main(_):
  env = AsyncVectorEnv(
    [
      bind(
        make_env,
        rank=i + 1,
        task=FLAGS.task,
        split=FLAGS.split,
        max_n_crossings=FLAGS.max_n_crossings,
        episode_length=FLAGS.episode_length,
      )
      for i in range(FLAGS.num_envs)
    ],
    autoreset_mode="SameStep",
  )

  per_ep_rewards = {i: [] for i in range(FLAGS.num_envs)}
  ep_rewards = []
  ep_lengths = []

  np.random.seed(FLAGS.seed)
  seeds = np.random.randint(0, 2**32 - 1, size=(FLAGS.num_envs,)).tolist()
  obs, info = env.reset(seed=seeds)

  while len(ep_rewards) < FLAGS.num_episodes:
    actions = env.action_space.sample()
    obs, rewards, terminations, truncations, info = env.step(actions)
    done = terminations | truncations
    for i in range(FLAGS.num_envs):
      per_ep_rewards[i].append(rewards[i])
      if done[i]:
        ep_rewards.append(sum(per_ep_rewards[i]))
        ep_lengths.append(len(per_ep_rewards[i]))
        per_ep_rewards[i] = []
  env.close()

  ep_rewards = np.array(ep_rewards)
  ep_lengths = np.array(ep_lengths)
  ep_successes = (ep_rewards >= 0.0).astype(float)

  logger.info("== Config:")
  logger.info(f"nxg: {FLAGS.max_n_crossings}")
  logger.info(f"Duration: {FLAGS.episode_length}")
  logger.info(f"Seed: {FLAGS.seed}")
  logger.info(f"Number of environments: {FLAGS.num_envs}")
  logger.info(f"Number of episodes: {len(ep_rewards)}")

  logger.info("== Results")
  logger.info(f"Average reward: {np.mean(ep_rewards):.2f}")
  logger.info(f"Average success rate: {np.mean(ep_successes):.2f}")
  logger.info(f"Average episode length: {np.mean(ep_lengths):.2f}")
  # success average length
  masked_lengths = ep_lengths[ep_successes.astype(bool)]
  if len(masked_lengths) > 0:
    logger.info(
      f"Average success episode length: {np.mean(masked_lengths):.2f}"
    )
  else:
    logger.info("No successful episodes.")

  # report
  output_dir = os.path.join(
    "results",
    "evaluations",
    "random_policy",
    f"seed_{FLAGS.seed}_{FLAGS.task}_split_{FLAGS.split}_nx_{FLAGS.max_n_crossings}",
  )
  report = dict(
    # checkpointing
    eval_slurm_job_id="random",
    load_from=None,
    command=" ".join(sys.argv),
    # tasks
    task=FLAGS.task,
    num_envs=FLAGS.num_envs,
    task_max_n_crossings=FLAGS.max_n_crossings,
    task_max_n_states=20,
    task_split=FLAGS.split,
    # results
    episode_success_rate=ep_successes.mean(),
    episode_rewards_mean=ep_rewards.mean(),
    episode_lengths_mean=ep_lengths.mean(),
    n_eval_episodes=len(ep_rewards),
    episode_rewards=ep_rewards.tolist(),
    episode_lengths=ep_lengths.tolist(),
    # aux
    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    seed=FLAGS.seed,
    slurm_job_id=os.getenv("SLURM_JOB_ID", "local"),
  )
  report_path = os.path.join(output_dir, "eval_report.json")
  os.makedirs(output_dir, exist_ok=False)
  with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
  logger.info(f"Eval report saved to {report_path}")


if __name__ == "__main__":
  app.run(main)
