import traci

def test_traci_connection():
    # Start SUMO with the TraCI interface
    traci.start(["sumo", "-c", "data/simulation_config.sumocfg"])

    # Run the simulation for a few steps
    for step in range(10):
        traci.simulationStep()

    traci.close()

if __name__ == "__main__":
    test_traci_connection()
