import traci
import numpy as np

class TrafficSignalEnv:
    def __init__(self):
        # Initialize the traffic signal environment with required variables
        self.num_vehicles = 0 
        self.action_space = 3
        self.state_space = 5

    def reset(self):
        # Reset the environment (traffic signal) to the starting state
        self.num_vehicles = 0
        traci.load()  # Load the SUMO simulation or reset simulation
        self.current_state = self.get_state()
        return self.current_state

    def step(self, action):
        self.change_traffic_signal(action)
        traci.simulationStep()

        next_state = self.get_state()

        reward = self.calculate_reward()

        done = False 

        return next_state, reward, done, {}

    def get_state(self):
        vehicle_count = traci.vehicle.getIDCount()
        queue_length = traci.lane.getLastStepHaltingNumber("lane_0")
        return np.array([vehicle_count, self.num_vehicles, queue_length])

    def change_traffic_signal(self, action):
        junctions = [f"junction_{i}" for i in range(9)]
        states = ["rrrrrrGGGGGG", "GGGGGGrrrrrr"]

        if action < 2:
            traci.trafficlight.setRedYellowGreenState(junctions[0], states[action])
        else:
            junction_index = (action - 2) // 2
            state_index = (action - 2) % 2
            traci.trafficlight.setRedYellowGreenState(junctions[junction_index], states[state_index])

    def calculate_reward(self):
        cars_finished = traci.simulation.getMinExpectedNumber()
        reward = -cars_finished
        return reward
