import traci
import time
import os


relative_path_to_cfg = "../data/grid.sumocfg"
path_to_cfg = os.path.abspath(relative_path_to_cfg)


def runSimulation():
    traci.start(["sumo", "-c", path_to_cfg])

    step = 0
    while step < 300:
        traci.simulationStep()
        step += 1
        
        print(f"{traci.simulation.getMinExpectedNumber()}")

    traci.close()


if __name__ == "__main__":
    runSimulation()

