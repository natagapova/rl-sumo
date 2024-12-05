import traci
import time
import os


relative_path_to_cfg = "../data/grid.sumocfg"
path_to_cfg = os.path.abspath(relative_path_to_cfg)


def runSimulation():
    traci.start(["sumo-gui", "-c", path_to_cfg])

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        time.sleep(0.1)
    
    traci.close()


if __name__ == "__main__":
    runSimulation()

