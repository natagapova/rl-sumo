import traci
import numpy as np
from typing import Tuple, List

class TrafficSignalEnv:
    def __init__(self):
        self.junction_ids = [f"junction_{i}" for i in range(9)]  # 9 junctions
        self.num_states = (6 * 4 + 4)  # Adjust based on your actual number of lanes per junction
        self.num_actions = 2  # Two possible phases for each junction
        self.max_steps = 1000
        self.current_step = 0

    def reset(self) -> np.ndarray:
        """Reset the environment and return initial state"""
        self.current_step = 0
        traci.load(["-c", "data/grid.sumocfg"])
        traci.simulation.step()
        return self._get_state()

    def step(self, actions: List[int]) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute action and return new state, reward, done flag, and info"""
        # Apply actions for each junction
        for junction_id, action in zip(self.junction_ids, actions):
            if action == 0:
                traci.trafficlight.setRedYellowGreenState(junction_id, "rrrrrrGGGGGG")
            else:
                traci.trafficlight.setRedYellowGreenState(junction_id, "GGGGGGrrrrrr")

        # Simulate one step
        traci.simulation.step()
        self.current_step += 1

        # Get new state and reward
        state = self._get_state()
        reward = self._get_reward()
        done = self.current_step >= self.max_steps

        return state, reward, done, {}

    def _get_state(self) -> np.ndarray:
        """Get comprehensive state representation for all junctions"""
        state = []
        for junction_id in self.junction_ids:
            # Get incoming lanes for this junction
            incoming_lanes = traci.trafficlight.getControlledLanes(junction_id)
            
            junction_state = []
            total_waiting_time = 0
            total_vehicles = 0
            
            for lane in incoming_lanes:
                # 1. Vehicle count and density
                lane_length = traci.lane.getLength(lane)
                vehicles = traci.lane.getLastStepVehicleNumber(lane)
                density = vehicles / (lane_length / 1000)  # vehicles per kilometer
                
                # 2. Queue information
                queue_length = traci.lane.getLastStepHaltingNumber(lane)
                queue_density = queue_length / (lane_length / 1000)
                
                # 3. Speed information
                mean_speed = traci.lane.getLastStepMeanSpeed(lane)
                max_speed = traci.lane.getMaxSpeed(lane)
                speed_ratio = mean_speed / max_speed if max_speed > 0 else 0
                
                # 4. Waiting times
                waiting_time = traci.lane.getWaitingTime(lane)
                total_waiting_time += waiting_time
                total_vehicles += vehicles
                
                # 5. Get vehicle-specific information
                vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
                if vehicle_ids:
                    vehicle_delays = []
                    vehicle_speeds = []
                    for vid in vehicle_ids:
                        # Calculate delay as difference between max and current speed
                        current_speed = traci.vehicle.getSpeed(vid)
                        vehicle_speeds.append(current_speed)
                        delay = max_speed - current_speed
                        vehicle_delays.append(delay)
                    
                    avg_vehicle_delay = np.mean(vehicle_delays) if vehicle_delays else 0
                    speed_variance = np.var(vehicle_speeds) if vehicle_speeds else 0
                else:
                    avg_vehicle_delay = 0
                    speed_variance = 0
                
                # Combine metrics for this lane
                lane_metrics = [
                    density / 50.0,  # Normalize density (assuming max 50 vehicles/km)
                    queue_density / 30.0,  # Normalize queue density
                    speed_ratio,  # Already normalized
                    waiting_time / 300.0,  # Normalize waiting time (max 300 seconds)
                    avg_vehicle_delay / max_speed,  # Normalize delay
                    speed_variance / (max_speed ** 2),  # Normalize variance
                ]
                
                junction_state.extend(lane_metrics)
            
            # Add junction-level metrics
            current_phase = traci.trafficlight.getPhase(junction_id)
            phase_duration = traci.trafficlight.getPhaseDuration(junction_id)
            time_since_last_change = traci.trafficlight.getNextSwitch(junction_id) - traci.simulation.getTime()
            
            junction_metrics = [
                current_phase / traci.trafficlight.getPhaseNumber(junction_id),  # Normalized phase
                time_since_last_change / phase_duration,  # Normalized time in phase
                total_waiting_time / (300.0 * len(incoming_lanes)),  # Normalized total waiting time
                total_vehicles / (20.0 * len(incoming_lanes))  # Normalized total vehicles
            ]
            
            junction_state.extend(junction_metrics)
            
            # Normalize and append to state
            state.extend(junction_state)
        
        return np.array(state)

    def _get_reward(self) -> float:
        """Calculate reward based on multiple metrics"""
        total_reward = 0
        
        for junction_id in self.junction_ids:
            incoming_lanes = traci.trafficlight.getControlledLanes(junction_id)
            
            # Penalties
            waiting_time_penalty = sum(traci.lane.getWaitingTime(lane) for lane in incoming_lanes)
            queue_length_penalty = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in incoming_lanes)
            
            # Rewards
            throughput_reward = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in incoming_lanes)
            avg_speed_reward = np.mean([traci.lane.getLastStepMeanSpeed(lane) for lane in incoming_lanes])
            
            # Combine rewards and penalties
            junction_reward = (
                -0.1 * waiting_time_penalty 
                - 0.2 * queue_length_penalty
                + 0.5 * throughput_reward
                + 1.0 * avg_speed_reward
            )
            
            total_reward += junction_reward
            
        return total_reward / len(self.junction_ids)  # Average reward across all junctions 