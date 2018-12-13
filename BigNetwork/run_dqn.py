import os, sys
sys.path.append(os.path.join('c:', os.sep, r'D:\InstalledProgram\SUMO\tools'))
os.environ['SUMO_HOME']= r'D:\InstalledProgram\SUMO'
import traci

#import argparse
#import gym
#from gym import wrappers
#import os.path as osp
import random
import numpy as np
import tensorflow as tf
#import tensorflow.contrib.layers as layers

import dqn
from dqn_utils import *
#from atari_wrappers import *


#def model(inp, num_phases, scope, reuse=False):
#    with tf.variable_scope(scope, reuse=reuse):
#        out = inp
#        out = layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.relu)
#        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
#        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
#        out = layers.fully_connected(out, num_outputs=num_phases, activation_fn=None)
#        return out

#def learn(session,num_timesteps,rew_file):
#    num_iterations = float(num_timesteps)
#
#    lr_multiplier = 10 #1.0
#    lr_schedule = PiecewiseSchedule([
#                                         (0,                   1e-4 * lr_multiplier),
#                                         (num_iterations / 10, 1e-4 * lr_multiplier),
#                                         (num_iterations / 2,  5e-5 * lr_multiplier),
#                                    ],
#                                    outside_value=5e-5 * lr_multiplier)
#    optimizer = dqn.OptimizerSpec(
#        constructor=tf.train.AdamOptimizer,
#        kwargs=dict(epsilon=1e-4),
#        lr_schedule=lr_schedule
#    )
#
#    exploration_schedule = PiecewiseSchedule(
#        [
#            (0, 1.0),
#            (300, 0.1),
#            (num_iterations / 2, 0.01),
#        ], outside_value=0.01
#    )
#
#    #dqnlearn(
#    dqnlearn(num_timesteps=num_timesteps,
#        num_hidden = 60,
#        optimizer_spec=optimizer,
#        session=session,
#        exploration=exploration_schedule,
#        replay_buffer_size= 200000,
#        batch_size=32,
#        gamma=0.9,
#        learning_starts=100,
#        learning_freq=4,
#        frame_history_len=1, # no frame history, we look only last data
#        target_update_freq=100,
#        grad_norm_clipping=10,
#        rew_file=rew_file,
#        double_q=True # True
#    )

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    session = tf.Session(config=tf_config)
    print("AVAILABLE GPUS: ", get_available_gpus())
    return session
    
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
    
def getobs(tl, num_phases, lanesEdge, OutLanesEdge): # need to have a shape to fit each lane where it should be
#    lsvn = np.zeros((4*4,),dtype=float)
#    lsms = np.zeros((4*4,),dtype=float)
#    rw = np.zeros((4*4,),dtype=float)
#    for i, edge in enumerate(lanesEdge):
#        for j, lane in enumerate(edge):
#           lsvn[4*i+j] = traci.lane.getLastStepVehicleNumber(lane)
#           lsms[4*i+j] = traci.lane.getLastStepMeanSpeed(lane)
#           rw[4*i+j] = lsvn[4*i+j]*(lsms[4*i+j]-traci.lane.getMaxSpeed(lane))
#    obs = np.concatenate([lsvn,lsms],axis=0)
#    return obs, np.sum(rw)

    lsvn = np.zeros((8*4,),dtype=float)
    lsms = np.zeros((8*4,),dtype=float)
    rw = np.zeros((4*4,),dtype=float)
    for i, edge in enumerate(lanesEdge):
        for j, lane in enumerate(edge):
           #print(i, j, lane)
           lsvn[4*i+j] = traci.lane.getLastStepVehicleNumber(lane)
           lsms[4*i+j] = traci.lane.getLastStepMeanSpeed(lane)
           rw[4*i+j] = lsvn[4*i+j]*(lsms[4*i+j]-traci.lane.getMaxSpeed(lane))
    #print()
    for i, edge in enumerate(OutLanesEdge):
        for j, lane in enumerate(edge):
           #print(i,j,lane)
           lsvn[4*i+j+16] = traci.lane.getLastStepVehicleNumber(lane)
           lsms[4*i+j+16] = traci.lane.getLastStepMeanSpeed(lane)
    obs = np.concatenate([lsvn,lsms],axis=0)
    return obs, np.sum(rw)

    
def dqnlearn(num_timesteps, *args, **kwargs):
    yellowTime = 3 # I have Warning: Vehicle '3.527' performs emergency stop at the end of lane for 3
    redTime = 0
    itBetweenDecision = int((yellowTime+redTime+3)/0.4) # each itteration is 0.4 s, I need at least 3s of green time
    
    lightList = traci.trafficlight.getIDList()
    obsLen = 2*4*4*2 # max of in and out lanes, 4 directions (only incoming edge are counted, so some have only 3 directions), max of 4 lanes per edges, 2 information per lane
    
    alg = []
    lanes = []
    lanesEdge = []
    OutLanes = []
    OutLanesEdge = []
    for i, tl in enumerate(lightList):
        phases = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl)[0].getPhases()
        num_phases = len(phases)//2 # The tl have 4 or 6 phases in their base program, so so it can be 2 or 3
        ph = []
        for phase in phases[::2]: # skip yellow
            ph.append(phase._phaseDef)
            
        outlane = set()
        #outlanedic = {}
        for link in traci.trafficlight.getControlledLinks(tl):
            #print('link', link[0][0]) # incoming lane
            #print('link', link[0][1]) # outgoing lane
            outlane.add(link[0][1]) # outgoing lane
            #print('link', link[0][2])  # junction lane
            #print()
            #outlanedic[link[0][0]] = link[0][1] # I want to ensure it's the straight path, not the left or right turning one though, so I will use another way
        OutLanes.append(outlane)
            
        # get number of edges and number of lanes per edges
        # they are sorted by their order in the phase
        # To be coherent: obs must be sorted in the same order as hidden and 
        # output. It works for obs since lanes are sorted in the same order as 
        # phase. Now, we need to ensure it works also for hidden state (since it is sent)
        lanes.append(traci.trafficlight.getControlledLanes(tl)) # also needed to get neighbors
        #print('lanes', lanes[-1])
        lanesEdge.append([])
        oldedge = None
        oldlane = None
        OutLanesEdge.append([])
        for lane in lanes[-1]: # 1st dim: tl, 2nd dim: one for each lane
            if lane == oldlane:
                continue
            oldlane = lane
            if traci.lane.getEdgeID(lane) == oldedge:
                lanesEdge[-1][-1].append(lane)
                #OutLanesEdge[-1].append(outlanedic[lane])
                #print(lane, outlanedic[lane])
            else:
                lanesEdge[-1].append([lane])
                #OutLanesEdge[-1].append([outlanedic[lane]])
                #print('new edge',lane, outlanedic[lane])
                oldedge = traci.lane.getEdgeID(lane)
                
        edgeDic = {}
        for lane in outlane: # it's a set so no repeat even though they aren't in the same order
            edge = traci.lane.getEdgeID(lane)
            if edge in edgeDic:
                OutLanesEdge[-1][edgeDic[edge]].append(lane)
                #print('old edge',lane, edgeDic[edge], edge)
            else:
                OutLanesEdge[-1].append([lane])
                edgeDic[edge] = len(OutLanesEdge[-1])-1
                #print('new edge',lane, edgeDic[edge], edge)
            
        print(lanesEdge[-1])
        print(OutLanesEdge[-1])
        alg.append(dqn.QLearner(num_phases = num_phases,observation_shape = obsLen,scope=str(i), phases=ph, timeBetweenDecision=itBetweenDecision*0.4, *args, **kwargs))
        obs, rw = getobs(tl, num_phases,lanesEdge[-1], OutLanesEdge[-1]) # first dim of lanesEdge is tl, 2nd dim is edge (direction), value is lane (3rd dim)
        alg[i].last_obs = obs
           
    neighTL = [] # I would like to know for sure which one is left and which one is right. Maybe use lanesEdge
    # Comparing lanes in common doesn't work, because only incoming lanes are in the list
    for i, tl in enumerate(lightList):
        neigh = np.array([-1,-1]) # only 2 neighbors in this map
        for j, tl2 in enumerate(lightList):
            if i != j:
#                if i == 0 and j == 4:
#                    print(lanes[i])
#                    print(OutLanes[j])
#                    print(lanes[i][:len(lanes[i])//2])
#                    print(lanes[i][len(lanes[i])//2:])
#                    print(set(lanes[i][:len(lanes[i])//2]).intersection(OutLanes[j]))
#                    print(set(lanes[i][len(lanes[i])//2:]).intersection(OutLanes[j]))
#                    print()
                if len(set(lanes[i][:len(lanes[i])//2]).intersection(OutLanes[j]))>0:
                   neigh[0] = j
                elif len(set(lanes[i][len(lanes[i])//2:]).intersection(OutLanes[j]))>0:
                   neigh[1] = j
        #print('neighbor tl', i, neigh, tl)
        neighTL.append(neigh) # 1st dim of neighTL is tl, value is the neighbor tl (2nd dim)
    # 0,4,5,1,3,2 # 0 didn't get 4
    neighTL[0][1] = 4 # correct manually. Due to edge expanding from 3 lanes to 4 lanes
    # I will need a better way on another map
    
    saver = tf.train.Saver() # create saver
    saveFrequency = 10000 
    saveFile = r"/model/A.ckpt"
    if False: # put True here if I want to restore
        saver.restore(kwargs['session'], saveFile)
        print("Model restored.")
        for i in range(len(alg)):
            alg[i].model_initialized = True # I believe it is initialized if it is restored
    
    episode_rewards = []
    action = np.empty_like(alg, int) #  Need to save all actions and then use them to learn each q
    reward = np.empty_like(alg, float)
    hid = np.empty((len(alg),kwargs['num_hidden']),float)
    Hs = np.empty((len(alg),2*kwargs['num_hidden']),float) # 2 because each tl has 2 neighbors
    #it = 0
    for it in range(num_timesteps):
        #print(it, traci.simulation.getTime())
        #if (it+1) % saveFrequency == 0: # save model # Don't save for now
        #    saver.save(kwargs['session'], saveFile)
        #    print("Model saved.")
        if traci.simulation.getMinExpectedNumber() <= 0:
            print('Error: No more cars')
            break
        if it%itBetweenDecision == 0:
            episode_rewards.append(reward/itBetweenDecision)
            for i, tl in enumerate(lightList):
                if it>itBetweenDecision: #not first time
                    done = False
                    #info = None
                    alg[i].replay_buffer.store_effect(idx, action[i], reward[i], done, Hs[i])
                    alg[i].update_model()
                
                lastObs = alg[i].last_obs/itBetweenDecision
                idx = alg[i].replay_buffer.store_frame(lastObs, 2*kwargs['num_hidden'])
                #lastObs = alg[i].replay_buffer.encode_recent_observation() # No need to encode, I'm not using context
        
                if not alg[i].model_initialized:
                    alg[i].model_initialized = True
                    initialize_interdependent_variables(alg[i].session, tf.global_variables(), {
                            alg[i].obs_t_ph: lastObs[np.newaxis,:],
                                                  })
                hid[i] = alg[i].session.run(alg[i].hid, feed_dict={alg[i].obs_t_ph:lastObs[np.newaxis,:]})
            
            alg[0].log_progress(episode_rewards, it) # need to log only once.
          
        # Choose action (with epsilon greedy)
            for i, tl in enumerate(lightList):
                h = []
                for ntl in neighTL[i]:
                    if ntl != -1:
                        h.append(hid[ntl,:])
                    else:
                        h.append(np.zeros((kwargs['num_hidden'],)))
                h = np.concatenate(h,axis=0)
                Hs[i] = h
                if random.random() > alg[i].exploration.value(alg[i].t):
                    q = alg[i].session.run(alg[i].q, feed_dict={alg[i].obs_t_ph:lastObs[np.newaxis,:],alg[i].tl_ph:h[np.newaxis,:]})
                    action[i] = np.argmax(q,axis=1)
                    #print('chosen action', action[i])
                else:
                    action[i] = alg[i].randomAction() #random.randrange(alg[i].num_phases) # random integer
                    #print('random action', action[i], alg[i].num_phases)
                alg[i].updatePhase(action[i])

                current_phase = traci.trafficlight.getRedYellowGreenState(tl)
                newphase = alg[i].phases[action[i]]
                #print(i, current_phase, newphase)
                if newphase != current_phase: # not same phase
                    traci.trafficlight.setPhaseDuration(tl, yellowTime) # trying to force switch by putting duration to 0. Actually, 0 made it swtich to red, so let's try yellowtime
                    #program = traci.trafficlight.Logic(programID, type, currentPhaseIndex, phases=None, subParameter=None)
                    yellowphase = list(current_phase)
                    #redphase = list(current_phase)
                    for ichar, char in enumerate(current_phase):
                        if newphase[ichar] == 'r' and char != 'r':
                            yellowphase[ichar] = 'y'
                            #redphase[ichar] = 'r'
                    yellowphase = ''.join(yellowphase)
                    #redphase = ''.join(redphase) # Says duration are in miliseconds, but that was in 2012 they might have changed it
                    phases = [traci.trafficlight.Phase(phaseDef=yellowphase,duration=yellowTime,duration1=yellowTime,duration2=yellowTime),
                              #traci.trafficlight.Phase(phaseDef=redphase,duration=redTime,duration1=redTime,duration2=redTime),
                              traci.trafficlight.Phase(phaseDef=newphase,duration=10000,duration1=10000,duration2=10000)]
                    program = traci.trafficlight.Logic(phases=phases, subID='0', type=0, subParameter=0, currentPhaseIndex=0)
                    traci.trafficlight.setCompleteRedYellowGreenDefinition(tl, program)
                    #print(traci.trafficlight.getRedYellowGreenState(tl))
                    # Problem: 1st time, It switches program, but getNextSwitch is the previous duration rather than new one
                    # It gets stuck in yellow phase
                
        traci.simulationStep()
        for i, tl in enumerate(lightList):
            #print('    ', i, traci.trafficlight.getRedYellowGreenState(tl), traci.trafficlight.getNextSwitch(tl), traci.trafficlight.getPhaseDuration(tl))
            obs, rw = getobs(tl, alg[i].num_phases,lanesEdge[i],OutLanesEdge[i])
            #print('obs', obs)
            #print('rw', rw)
            if it%itBetweenDecision == 0:
                alg[i].last_obs = obs
                reward[i] = rw
            else:
                alg[i].last_obs += obs
                reward[i] += rw
        
    traci.close()
    sys.stdout.flush()

def main():
    num_timesteps = 50000 # It's actually not the number I get, look to that
    num_iterations = float(num_timesteps)
    printMyRoute(num_timesteps)
    #sumoBinary = r"D:\InstalledProgram\SUMO\bin\sumo-gui.exe"
    sumoBinary = r"D:\InstalledProgram\SUMO\bin\sumo.exe"
        
    # Run training
    f = open('triedParameters.txt','a+')
    ntries = 100
    for ntry in range(ntries):
        # hyperparameter random search
        lr_multiplier = 10**random.uniform(-2, 2) #1.0
        num_hidden = random.randint(30,100)
        gamma = random.uniform(0.8, 0.95) # horizon should be 1/(1-gamma). We are getting reward every 6 seconds, so 0.9 for 60 seconds should be good
        learning_freq = random.randint(1,20)
        target_update_freq = random.randint(10,200)
        explor1 = random.randint(100,1000)
        explor2 = int(num_iterations*random.uniform(0.1,0.9))
        if explor2 < explor1:
            explor2 = (num_iterations+explor1)/2
        print(ntry, lr_multiplier, num_hidden, gamma, learning_freq, target_update_freq, explor1, explor2, file=f)
        
        traci.start([sumoBinary, "-c", "conf2.sumocfg","--tripinfo-output", "tripinfo"+str(ntry)+".xml","--no-step-log","--time-to-teleport", "-1"])
        seed = random.randint(0, 9999)
        print('random seed = %d' % seed)
        session = get_session()
        rew_file='reward'+str(ntry)+'.pkl'
    
        lr_schedule = PiecewiseSchedule([
                                             (0,                   1e-4 * lr_multiplier),
                                             (num_iterations / 10, 1e-4 * lr_multiplier),
                                             (num_iterations / 2,  5e-5 * lr_multiplier),
                                        ],
                                        outside_value=5e-5 * lr_multiplier)
        optimizer = dqn.OptimizerSpec(
            constructor=tf.train.AdamOptimizer,
            kwargs=dict(epsilon=1e-4),
            lr_schedule=lr_schedule
        )
    
        exploration_schedule = PiecewiseSchedule(
            [
                (0, 1.0),
                (explor1, 0.1),
                (explor2, 0.01),
            ], outside_value=0. # No exploration towards the end so we can see the true reward
        )
    
        #dqnlearn(
        dqnlearn(num_timesteps=num_timesteps,
            num_hidden = num_hidden,
            optimizer_spec=optimizer,
            session=session,
            exploration=exploration_schedule,
            replay_buffer_size= 200000,
            batch_size=32,
            gamma=gamma,
            learning_starts=100,
            learning_freq=learning_freq,
            frame_history_len=1, # no frame history, we look only last data
            target_update_freq=target_update_freq,
            grad_norm_clipping=10,
            rew_file=rew_file,
            double_q=True # True
        )

if __name__ == "__main__":
    main()
