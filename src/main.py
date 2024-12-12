import traci
import os
import numpy as np
from rl_agent import RLAgent
from situation import TrafficSignalEnv

def main():
    # Initialize SUMO simulation
    relative_path_to_cfg = "../data/grid.sumocfg" # Remove "../" if running from root
    path_to_cfg = os.path.abspath(relative_path_to_cfg)
    sumo_cmd = ["sumo", "-c", path_to_cfg, "--no-step-log", "--no-warnings", "--no-internal-links", "--step-length", "0.1"]
    traci.start(sumo_cmd)

    # Initialize environment and agent
    env = TrafficSignalEnv()
    state_dim = env.num_states  # Total state dimensions for all junctions
    action_dim = env.num_actions  # Now 5 actions instead of 2
    agent = RLAgent(state_dim=state_dim, action_dim=action_dim, num_junctions=9)

    # Training parameters
    num_episodes = 1000
    update_target_every = 5
    save_model_every = 100
    action_interval = 10  # Apply actions every 10 steps
    
    # Add logging for phase durations
    episode_durations = []

    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        last_action_step = 0  # Track when we last took an action
        current_actions = None  # Store current actions

        while not done:
            # Only take new actions every action_interval steps
            if step - last_action_step >= action_interval or current_actions is None:
                current_actions = agent.act(state)
                last_action_step = step
                
            next_state, reward, done, _ = env.step(current_actions)
            
            # Only store experiences when we take new actions
            if step - last_action_step == 0:
                agent.remember(state, current_actions, reward, next_state, done)
                agent.learn()
            
            state = next_state
            total_reward += reward
            step += 1

        # Log average phase durations for this episode
        avg_duration = np.mean([env.phase_durations[j] for j in env.junction_ids])
        episode_durations.append(avg_duration)

        if episode % update_target_every == 0:
            agent.update_target_model()

        if episode % save_model_every == 0:
            if not os.path.exists("../models"):
                os.makedirs("../models")
            agent.save_model(f"../models/agent_episode_{episode}.h5")

        print(f"Episode: {episode}, Total Reward: {total_reward}, Steps: {step}, "
              f"Avg Phase Duration: {avg_duration:.1f}s, Epsilon: {agent.epsilon:.3f}")

    traci.close()

if __name__ == "__main__":
    main()
