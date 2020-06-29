import numpy as np
import random
import copy

import torch
import torch.nn.functional as F

GAMMA = 0.99 # discount factor
TAU = 1e-3 # for soft update of target parameters
UPDATE_EVERY = 5
NOISE_MULTIPLIER = 2

class Agent():
	"""Interacts with and learns from the environment."""
		
	def __init__(self, action_size, random_seed, batch_size, actor_local, actor_target, actor_optimizer, critic_local, critic_target, critic_optimizer, memory, noise, device='cpu'):
		"""Initialize an Agent object.
			
		Params
		======
			action_size (int): dimension of each action
			random_seed (int): random seed
			batch_size (int): minibatch size for learning
			actor_local (Actor): local actor network
			actor_target (Actor): target actor network
			actor_optimizer (Optimizer): actor optimizer
			critic_local (Critic): local critic network
			critic_target (Critic): target critic network
			critic_optimizer (Optimizer): critic optimizer
			memory (ReplayBuffer): replay buffer
			noise (OUNoise): noise generator
			device (string): device to use
		"""
		self.action_size = action_size
		self.seed = random.seed(random_seed)
		self.batch_size = batch_size
		self.actor_local = actor_local
		self.actor_target = actor_target
		self.actor_optimizer = actor_optimizer
		self.critic_local = critic_local
		self.critic_target = critic_target
		self.critic_optimizer = critic_optimizer
		self.memory = memory
		self.noise = noise
		self.device = device
		self.t_step = 0
		
	def step(self, state, action, reward, next_state, done):
		"""Save experience in replay memory, and use random sample from buffer to learn.
		Params
		======
			state (numpy.ndarray[float]): current environment state
			action (numpy.ndarray[float]): action to take
			reward (float): reward for given action in given state
			next_state (numpy.ndarray[float]): environment state after the given action is taken
			done (bool): whether the episode has terminated
		"""
		# Save experience / reward
		self.memory.add(state, action, reward, next_state, done)

		self.t_step = (self.t_step + 1) % UPDATE_EVERY
			
		# Learn, if enough samples are available in memory
		if self.t_step == 0 and len(self.memory) > self.batch_size:
			experiences = self.memory.sample()
			self.learn(experiences, GAMMA)

	def act(self, state, eps, add_noise=True):
		"""Returns actions for given state as per current policy.
		Params
		======
			state (numpy.ndarray[float]): current environment state
			eps (float): epsilon value for noise generation and random action selection
			add_noise (bool): whether or not to add noise
		"""
		if random.random() < eps:
			return np.random.randn(self.action_size)
		else:
			state = torch.from_numpy(state).float().to(self.device)
			self.actor_local.eval()
			with torch.no_grad():
				action = self.actor_local(state).cpu().data.numpy()
			self.actor_local.train()
			if add_noise:
				action += self.noise.sample() * eps * NOISE_MULTIPLIER
			return np.clip(action, -1, 1)

	def reset(self):
		"""Reset noise."""
		self.noise.reset()

	def learn(self, experiences, gamma):
		"""Update policy and value parameters using given batch of experience tuples.
		Q_targets = r + γ * self.critic_target(next_state, actor_target(next_state))
		where:
			actor_target(state) -> action
			self.critic_target(state, action) -> Q-value
		Params
		======
			experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
			gamma (float): discount factor
		"""
		states, actions, rewards, next_states, dones = experiences

		# ---------------------------- update critic ---------------------------- #
		# Get predicted next-state actions and Q values from target models
		actions_next = self.actor_target(next_states)
		Q_targets_next = self.critic_target(next_states, actions_next)
		# Compute Q targets for current states (y_i)
		Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
		# Compute critic loss
		Q_expected = self.critic_local(states, actions)
		critic_loss = F.mse_loss(Q_expected, Q_targets)
		# Minimize the loss
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
		self.critic_optimizer.step()

		# ---------------------------- update actor ---------------------------- #
		# Compute actor loss
		actions_pred = self.actor_local(states)
		actor_loss = -self.critic_local(states, actions_pred).mean()
		# Minimize the loss
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# ----------------------- update target networks ----------------------- #
		self.soft_update(self.critic_local, self.critic_target, TAU)
		self.soft_update(self.actor_local, self.actor_target, TAU)                     

	def soft_update(self, local_model, target_model, tau):
		"""Soft update model parameters.
		θ_target = τ*θ_local + (1 - τ)*θ_target
		Params
		======
			local_model: PyTorch model (weights will be copied from)
			target_model: PyTorch model (weights will be copied to)
			tau (float): interpolation parameter 
		"""
		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)