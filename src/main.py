import traci
import time
from rl_agent import RLAgent
from situation import TrafficSignalEnv

def main():
    # Initialize SUMO simulation
    sumo_cmd = ["sumo", "-n", "your_network_file.net.xml", "--start"]
    traci.start(sumo_cmd)

    # Initialize the reinforcement learning agent
    agent = RLAgent(state_dim=5, action_dim=3)
    env = TrafficSignalEnv() 

    # Training loop
    for episode in range(1000):  # Example: 1000 training episodes
        state = env.reset() 
        done = False

        while not done:
            action = agent.act(state) 
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state) 
            state = next_state

            if episode % 10 == 0:
                print(f"Episode: {episode}, Reward: {reward}")

        if episode % 100 == 0:
            agent.save_model(f"agent_{episode}.h5")
        
    # Stop the simulation
    traci.close()

if __name__ == "__main__":
    main()
