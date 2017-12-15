"""Reinforce with linear baseline and MLP policy."""

import gym
import numpy as np
import tensorflow as tf


class MLPPolicy(object):
  def __init__(self, input_shape, num_actions, learning_rate=0.01):
    self.observations = tf.placeholder(tf.float32, [None, *input_shape])
    self.rewards = tf.placeholder(tf.float32, shape=None)
    self.actions = tf.placeholder(tf.int32, shape=None)

    observations = tf.reshape(
      self.observations, [-1, np.prod(input_shape)])

    net = tf.layers.dense(observations, 64, activation=tf.nn.relu)
    net = tf.layers.dense(net, 64, activation=tf.nn.relu)
    logits = tf.layers.dense(net, num_actions, activation=None)
    value_func = tf.layers.dense(net, 1, activation=None)
    self.probs = tf.nn.softmax(logits)
    log_probs = tf.nn.log_softmax(logits)

    actions_one_hot = tf.one_hot(self.actions, num_actions)
    masked_log_probs = tf.reduce_sum(actions_one_hot * log_probs, axis=1)

    td_error = self.rewards - tf.reshape(value_func, (-1,))
    pg_error = - masked_log_probs * tf.stop_gradient(td_error)
    vf_error = 0.5 * td_error ** 2

    self.loss = tf.reduce_mean(pg_error + vf_error)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    self.train_op = optimizer.minimize(self.loss)

  def get_action(self, session, observation):
    probs = session.run(self.probs, {self.observations: observation})
    return categorical_sample(probs)


def categorical_sample(prob_vector):
  cum_probs = np.cumsum(prob_vector)
  return np.argmax(cum_probs > np.random.random())


def get_traj(env, session, policy, max_steps=1000):
  observations = []
  rewards = []
  actions = []
  obs = env.reset()
  for i in range(max_steps):
    observations.append(obs)
    action = policy.get_action(session, [obs])
    obs, rew, done, _ = env.step(action)
    actions.append(action)
    rewards.append(rew)
    if done:
      break

  return observations, actions, rewards


def discount_rewards(rewards, gamma=0.99):
  for i in reversed(range(len(rewards) - 1)):
    rewards[i] += rewards[i + 1] * gamma
  return rewards


def train(env, session, policy, num_iterations=1000, num_trajs_per_batch=30):
  for i in range(num_iterations):
    observations = []
    rewards = []
    actions = []
    all_rewards = 0.0
    all_lengths = []
    for j in range(num_trajs_per_batch):
      obs, acts, rews = get_traj(env, session, policy)
      all_rewards += np.sum(rews)
      observations.extend(obs)
      actions.extend(acts)
      rewards.extend(discount_rewards(rews))
      all_lengths.append(len(obs))

    loss, _ = session.run([policy.loss, policy.train_op],
                          {policy.observations:observations,
                           policy.rewards: rewards,
                           policy.actions: actions})
    print('Loss: {} Rews: {} Lengths: {}'.format(
        loss, all_rewards, np.mean(all_lengths)))


def main():
  session = tf.Session()
  env = gym.make('CartPole-v0')
  policy = MLPPolicy(env.observation_space.shape, env.action_space.n)
  session.run(tf.global_variables_initializer())
  train(env, session, policy)


if __name__ == '__main__':
  main()