import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class RLAgent:
    def __init__(self, state_dim, action_dim, num_junctions):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_junctions = num_junctions

        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.memory = deque(maxlen=100000)

        self.policy_model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # Use a smaller network with fewer parameters to improve training speed
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_dim, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_dim, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.policy_model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_dim, size=self.num_junctions)
        q_values = self.policy_model.predict(state)
        return np.argmax(q_values, axis=1)  # Parallel action selection for all junctions

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, targets = [], []

        for state, action, reward, next_state, done in batch:
            target = self.policy_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                q_future = np.amax(self.target_model.predict(next_state)[0])
                target[0][action] = reward + self.gamma * q_future

            states.append(state)
            targets.append(target)

        self.policy_model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, file_path):
        self.policy_model.save(file_path)

    def load_model(self, file_path):
        self.policy_model.load_weights(file_path)
        self.update_target_model()
