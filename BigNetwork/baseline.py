import os, sys
sys.path.append(os.path.join('c:', os.sep, r'D:\InstalledProgram\SUMO\tools'))
os.environ['SUMO_HOME']= r'D:\InstalledProgram\SUMO'
import traci

import random
import pickle
#import numpy as np
    
def printMyRoute(num_timesteps):
    # In net.net.xml: 1,2,3,4,6,7,9,10,12 are ingoing
    # 5, 7, 8, 10, 11, 13, 14, 15, 16 are outgoing
    # 17,18,19,20 are pedestrian
    ingoing = [1,2,3,4,6,8,10,12,14]
    outgoing = [5, 7, 9,11,13,15,16,17,18]
    main = {1,9,10,18}
    easyblocked = {2,4}
#    pedestrian = [19,20,21,22]
    f = open('myRoute.xml','w')
    print('<routes>', file=f)
    id = 0
    for node1 in ingoing:
        for node2 in outgoing:
            if node1 != node2:
                if node1 in main and node2 in main:
                    probability = 0.1
                elif node1 in main or node2 in main:
                    probability = 0.01
                else:
                    probability = 0.005
                if node1 in easyblocked:
                    probability = 0.002
                id += 1
                print('<flow id="{4}" begin="0" probability="{0}" end="{1}" from="{2}" to="{3}"/>'.format(str(probability),str(num_timesteps/0.4),str(node1),str(node2),str(id)), file=f)
#    for node1 in pedestrian:
#        for node2 in pedestrian:
#            if node1 != node2:
#                id += 1
#                print('<flow id="{4}" begin="0" probability="{0}" end="{1}" type="DEFAULT_PEDTYPE" from="{2}" to="{3}"/>'.format(str(probability),str(num_timesteps/0.4),str(node1),str(node2),str(id)), file=f)
    print('</routes>', file=f)
    f.close()
    
def main():
    num_timesteps = 3000
    printMyRoute(num_timesteps)
    sumoBinary = r"D:\InstalledProgram\SUMO\bin\sumo-gui.exe"
    #sumoBinary = r"D:\InstalledProgram\SUMO\bin\sumo.exe"
    traci.start([sumoBinary, "-c", "conf2.sumocfg","--tripinfo-output", "Baselinetripinfo.xml","--no-step-log"])
        
    # Run training
    seed = random.randint(0, 9999)
    print('random seed = %d' % seed)
    
    lightList = traci.trafficlight.getIDList()
    lanes = []
    for i, tl in enumerate(lightList):
        lanes.append(traci.trafficlight.getControlledLanes(tl))
        
        # Need to synchronize the traffic lights to be a descent baseline
        # Turn right from 2, turn right and left from 3, turn left from 4are full because they don't have room when they
        # have green.
        # need to change the phase so that it's green when they want to turn
        # 60(m) 3 24(s) 3
        # 9(t) 3 12(s) 3 60(m) 3
        # 60-9-3-12-3=33 of the main remaining. That's probably enough
        # Must always be m when the center junction is t or s, but we need main-main too.
        # Offset of 63 to start with turn
        # Maybe try without protected turn
    
    # In run_dqn.py:
    # episode_rewards: list of reward/itBetweenDecision for each timestep where there is a decision
    # reward: list of reward (sum of reward for each line) for each traffic light
    
    traci.trafficlight.setPhase('cluster_1918039897_42445511_5829753046_5829753047_5829779530_5829779533_5829927765_5829927770',
                                3)
    
    episode_rewards = []
    for it in range(num_timesteps):
        if traci.simulation.getMinExpectedNumber() <= 0:
            print('Error: No more cars')
            break
        traci.simulationStep()
        rewards = []
        for i, tl in enumerate(lightList):
            rewards.append(0)
            for lane in lanes[i]:
                rewards[-1] += traci.lane.getLastStepVehicleNumber(lane)*traci.lane.getLastStepMeanSpeed(lane)-traci.lane.getMaxSpeed(lane)
        episode_rewards.append(rewards)
        
        
    traci.close()
    with open('baseline_reward.pkl', 'wb') as f:
        pickle.dump(episode_rewards, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
