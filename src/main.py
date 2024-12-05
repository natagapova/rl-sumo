import traci
import numpy as np
from rl_agent import RLAgent
from situation import TrafficSignalEnv

def main():
    # Initialize SUMO simulation
    sumo_cmd = ["sumo", "-c", "data/grid.sumocfg"]
    traci.start(sumo_cmd)

    # Initialize environment and agent
    env = TrafficSignalEnv()
    state_dim = env.num_states * 9  # Total state dimensions for all junctions
    action_dim = env.num_actions
    agent = RLAgent(state_dim=state_dim, action_dim=action_dim)

    # Training parameters
    num_episodes = 1000
    update_target_every = 5
    save_model_every = 100

    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0

        while not done:
            # Get actions for all junctions
            actions = agent.act(state)
            
            # Execute actions and get new state
            next_state, reward, done, _ = env.step(actions)
            
            # Store experience in memory
            agent.remember(state, actions, reward, next_state, done)
            
            # Learn from experience
            agent.learn()
            
            state = next_state
            total_reward += reward
            step += 1

        # Update target network periodically
        if episode % update_target_every == 0:
            agent.update_target_model()

        # Save model periodically
        if episode % save_model_every == 0:
            agent.save_model(f"models/agent_episode_{episode}.h5")

        print(f"Episode: {episode}, Total Reward: {total_reward}, Steps: {step}, Epsilon: {agent.epsilon:.3f}")

    traci.close()

if __name__ == "__main__":
    main()
