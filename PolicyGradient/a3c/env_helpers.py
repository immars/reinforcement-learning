
import numpy as np
import logging
import gym
from lib.atari import helpers as atari_helpers


def make_env(env_name, wrap=True):
  env = gym.envs.make(env_name)
  # remove the timelimitwrapper
  if wrap:
    env_class = env.env.__class__.__name__
    print("env_class:", env_class)
    if env_class == "AtariEnv":
      env = env.env
      env = atari_helpers.AtariEnvWrapper(env)
    elif env_class == "CarRacing":
      env = CarEnvWrapper(env, 5, 5)
  return env


class CarEnvWrapper(object):
  """
  Wraps an Atari environment to end an episode when a life is lost.
  """
  def __init__(self, env, steer_n, speed_n):
    self.steer_n, self.speed_n = steer_n, speed_n
    self.env = env
    self.action_n = steer_n * speed_n

  def __getattr__(self, name):
    print("getattr:", name)
    return getattr(self.env, name)

  def step(self, *args, **kwargs):
    # lives_before = self.env.ale.lives()
    if len(args) > 0:
        action_i = args[0]
    else:
        action_i = kwargs["action"]
    action_c = self.action_d2c(action_i)
    logging.warning("action d2c: %s => %s", action_i, action_c)
    next_state, reward, done, info = self.env.step(action_c)
    # lives_after = self.env.ale.lives()
    #
    # # End the episode when a life is lost
    # if lives_before > lives_after:
    #   done = True
    #
    # # Clip rewards to [-1,1]
    # reward = max(min(reward, 1), -1)

    return next_state, reward, done, info

  def action_c2d(self, action):
    """
    continuous action to discrete action
    :param action:
    :return:
    """
    steer_i = int((action[0] - (-1.0)) / 2.0 * self.steer_n)
    steer_i = self.steer_n - 1 if steer_i >= self.steer_n else steer_i
    if abs(action[1]) > abs(action[2]):
        speed_action = action[1]
    else:
        speed_action = -action[2]
    speed_i = int((speed_action - (-1.0)) / 2.0 * self.speed_n)
    speed_i = self.speed_n - 1 if speed_i >= self.speed_n else speed_i
    return steer_i * self.speed_n + speed_i

  def action_d2c(self, action):
    steer_i = int(action / self.speed_n)
    speed_i = action % self.speed_n
    action_c = np.asarray([0., 0., 0.])
    action_c[0] = float(steer_i) / self.steer_n * 2 - 1.0 + 1.0 / self.steer_n
    speed_c = float(speed_i) / self.speed_n * 2 - 1.0 + 1.0 / self.speed_n
    if speed_c >= 0:
      action_c[1], action[2] = speed_c, 0
    else:
      action_c[1], action[2] = 0, -speed_c
    return action_c
