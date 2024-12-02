import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers

class RLAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        
        # Experience replay buffer
        self.memory = deque(maxlen=2000)
        
        # Build the neural network model
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(layers.InputLayer(input_shape=(self.state_dim,)))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_dim, activation='linear'))  # Output: one value per action
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state = np.reshape(state, [1, self.state_dim])
        return np.argmax(self.model.predict(state)[0])

    def learn(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

        # Learn from the experience
        if len(self.memory) < 32:
            return

        batch = random.sample(self.memory, 32)
        for state, action, reward, next_state in batch:
            state = np.reshape(state, [1, self.state_dim])
            next_state = np.reshape(next_state, [1, self.state_dim])

            target = reward + self.gamma * np.max(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target

            # Train the model
            self.model.fit(state, target_f, epochs=1, verbose=0)

        # Decay epsilon (exploration vs exploitation)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filename):
        self.model.save(filename)

    def load_model(self, filename):
        self.model = tf.keras.models.load_model(filename)
