import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import random
from collections import deque
from typing import List

class RLAgent:
    def __init__(self, state_dim: int, action_dim: int, num_junctions: int = 9):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_junctions = num_junctions
        self.memory = deque(maxlen=100000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self) -> Model:
        """Build neural network model"""
        model = Sequential([
            Input(shape=(self.state_dim,)),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(self.action_dim * self.num_junctions, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def update_target_model(self):
        """Update target model weights"""
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state: np.ndarray) -> List[int]:
        """Choose action(s) based on epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return [random.randrange(self.action_dim) for _ in range(self.num_junctions)]
        
        state = np.reshape(state, [1, -1])
        act_values = self.model.predict(state, verbose=0)
        act_values = np.reshape(act_values, (self.num_junctions, self.action_dim))
        return [np.argmax(act_values[i]) for i in range(self.num_junctions)]

    def remember(self, state: np.ndarray, action: List[int], reward: float, 
                next_state: np.ndarray, done: bool):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        """Train the model using experience replay"""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])

        # Predict Q-values for current states
        targets = self.model.predict(states, verbose=0)
        
        # Predict Q-values for next states using target network
        next_q_values = self.target_model.predict(next_states, verbose=0)

        for i in range(self.batch_size):
            for j in range(self.num_junctions):
                if dones[i]:
                    targets[i][j * self.action_dim + actions[i][j]] = rewards[i]
                else:
                    targets[i][j * self.action_dim + actions[i][j]] = rewards[i] + \
                        self.gamma * np.max(next_q_values[i][j * self.action_dim:(j + 1) * self.action_dim])

        self.model.fit(states, targets, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filepath: str):
        """Save the model weights"""
        self.model.save_weights(filepath)

    def load_model(self, filepath: str):
        """Load the model weights"""
        self.model.load_weights(filepath)
