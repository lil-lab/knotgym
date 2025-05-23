import unittest

import gymnasium as gym
import numpy as np

import knotgym  # noqa: F401


class TestEnvs(unittest.TestCase):
  def test_make(self):
    for task in ("unknot", "tie", "eq1"):
      env = gym.make("knotgym/Unknot-v0", task=task)
      env.close()
      self.assertTrue(True)

  def test_timeout(self):
    max_episode_steps = 5
    env = gym.make(
      "knotgym/Unknot-v0", task="unknot", duration=max_episode_steps
    )
    obs, info = env.reset()
    for _ in range(max_episode_steps):
      obs, reward, terminated, truncated, info = env.step(
        np.zeros(env.action_space.shape)
      )
    env.close()
    self.assertFalse(terminated)
    self.assertTrue(truncated)
