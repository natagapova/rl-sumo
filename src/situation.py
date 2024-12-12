import numpy as np
import traci

class TrafficSignalEnv:
    def __init__(self):
        self.junction_ids = traci.trafficlight.getIDList()
        self.num_junctions = len(self.junction_ids)
        self.num_states = 4 * self.num_junctions  # Queue lengths for each direction
        self.num_actions = 5  # Added new phase options

        self.phase_durations = {j: [] for j in self.junction_ids}

    def reset(self):
        traci.load("../data/grid.sumocfg")
        traci.simulationStep()
        return self._get_state()

    def step(self, actions):
        for junction, action in zip(self.junction_ids, actions):
            traci.trafficlight.setPhase(junction, action)

        for _ in range(10):  # Steps per action
            traci.simulationStep()

        next_state = self._get_state()
        reward = self._compute_reward()
        done = traci.simulation.getMinExpectedNumber() <= 0

        return next_state, reward, done, {}

    def _get_state(self):
        state = []
        for junction in self.junction_ids:
            queues = [traci.edge.getLastStepVehicleNumber(edge) for edge in traci.trafficlight.getControlledLinks(junction)]
            state.extend(queues)
        return np.array(state)
    
    def _compute_reward(self):
        rewards = []
        for junction in self.junction_ids:
            queue_lengths = [traci.edge.getLastStepVehicleNumber(edge) for edge in traci.trafficlight.getControlledLinks(junction)]
            rewards.append(-sum(queue_lengths))  # Negative for minimizing queues
        return np.mean(rewards)

    def close(self):
        traci.close()