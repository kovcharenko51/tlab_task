import gym
import torch
import numpy as np
import torch.nn as nn
from typing import Union


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


@torch.no_grad()
def eval_actor_final(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int
) -> np.ndarray:
    seeds = np.random.randint(0, 1337, size=10)
    final_scores = []
    for seed in seeds:
        env.seed(seed)
        actor.eval()
        episode_rewards = []
        for _ in range(n_episodes):
            state, done = env.reset(), False
            episode_reward = 0.0
            while not done:
                action = actor.act(state, device)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
            episode_rewards.append(episode_reward)
        eval_scores = np.as_array(episode_rewards)
        eval_score = eval_scores.mean()
        normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
        final_scores.append(normalized_eval_score)
    actor.train()
    return np.mean(final_scores)