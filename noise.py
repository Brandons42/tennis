import numpy as np
import random
import copy

class OUNoise:
	"""Ornstein-Uhlenbeck process."""

	def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
		"""Initialize parameters and noise process.
		Params
		======
			size (int): number of elements noise must be applied to / action size
			seed (int): random seed to be used
			mu (float): mean
			theta (float): constant determining reversion to mean
			sigma (float): volatility constant
		"""
		self.mu = mu * np.ones(size)
		self.theta = theta
		self.sigma = sigma
		self.seed = random.seed(seed)
		self.reset()

	def reset(self):
		"""Reset the internal state (= noise) to mean (mu)."""
		self.state = copy.copy(self.mu)

	def sample(self):
		"""Update internal state and return it as a noise sample."""
		x = self.state
		dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
		self.state = x + dx
		return self.state