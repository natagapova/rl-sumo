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
        return np.array([vehicle_count, self.num_vehicles])

    def change_traffic_signal(self, action):
        if action == 0:
            traci.trafficlight.setRedYellowGreenState("junction_0", "rrrrrrGGGGGG")
        elif action == 1:
            traci.trafficlight.setRedYellowGreenState("junction_0", "GGGGGGrrrrrr")
        else:
            traci.trafficlight.setRedYellowGreenState("junction_0", "rrrrrGGGGGG")

    def calculate_reward(self):
        # Example reward calculation (you can customize this)
        queue_length = traci.lane.getLastStepHaltingNumber("lane_0")
        reward = -queue_length
        return reward
