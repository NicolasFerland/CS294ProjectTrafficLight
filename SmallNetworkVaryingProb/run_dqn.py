import os, sys
sys.path.append(os.path.join('c:', os.sep, r'D:\InstalledProgram\SUMO\tools'))
os.environ['SUMO_HOME']= r'D:\InstalledProgram\SUMO'
import traci
import random
import numpy as np
import tensorflow as tf
import dqn
from dqn_utils import *


stepLength = 1 # step of the Sumo simulation. Originally I used 0.4. 

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
    
def printMyRoute(num_timesteps, isim):
    # In net.net.xml: 1,2,3,4,6,7,9,10,12 are ingoing
    # 5, 7, 8, 10, 11, 13, 14, 15, 16 are outgoing
    # 17,18,19,20 are pedestrian
    ingoing = [1,2,3,4]
    outgoing = [5,6,7,8]
    main = {1,3,5,7}
    side = {2,4,6,8}
#    pedestrian = [19,20,21,22]   
    g = open('RouteSimulations.txt','a+') # keep history of all simulations 
    print(isim, file=g)
    
    f = open('myRoute.xml','w')
    print('<routes>', file=f)
    id = 0
    for i1, node1 in enumerate(ingoing):
        for i2, node2 in enumerate(outgoing):
                # 0.08, 0.06, 0.04, 0.002 is heavy traffic 
                if node1 in main and node2 in main:
                    probability = 0.06
                elif node1 in side or node2 in side:
                    probability = 0.04
                else:
                    probability = 0.02
                if i1 == i2: # u-turn
                    probability = 0.002
                id += 1
                probability = np.random.normal(probability,probability/3)
                probability = max(probability,0.0001)
                print(node1, node2, probability, file=g)
                print('<flow id="{4}" begin="0" probability="{0}" end="{1}" from="{2}" to="{3}"/>'.format(str(probability),str(num_timesteps/stepLength),str(node1),str(node2),str(id)), file=f)
#    for node1 in pedestrian:
#        for node2 in pedestrian:
#            if node1 != node2:
#                id += 1
#                print('<flow id="{4}" begin="0" probability="{0}" end="{1}" type="DEFAULT_PEDTYPE" from="{2}" to="{3}"/>'.format(str(probability),str(num_timesteps/stepLength),str(node1),str(node2),str(id)), file=f)
    print('</routes>', file=f)
    print('', file=g)
    f.close()
    g.close()
    
#def getobs(tl, num_phases, lanesEdge, OutLanesEdge):
def getobs(tl, num_phases, lanesEdge): 
    nddir = 4
    nblane = len(lanesEdge[0]) # in general, do max
    #nboutlane = len(OutLanesEdge[0]) # in general, do max

    lsvn = np.zeros((nddir*nblane,),dtype=float)
    lsms = np.zeros((nddir*nblane,),dtype=float)
    lswt = np.zeros((nddir*nblane,),dtype=float)
    lshn = np.zeros((nddir*nblane,),dtype=float)
    lsmw = np.zeros((nddir*nblane,),dtype=float)
    rw = np.zeros((nddir*nblane,),dtype=float)
    dhead = np.zeros((nddir*nblane,),dtype=float)
    for i, edge in enumerate(lanesEdge):
        for j, lane in enumerate(edge):
           #print(i, j, lane)
           lsvn[nblane*i+j] = traci.lane.getLastStepVehicleNumber(lane)
           lsms[nblane*i+j] = traci.lane.getLastStepMeanSpeed(lane)
           lswt[nblane*i+j] = traci.lane.getWaitingTime(lane)
           lshn[nblane*i+j] = traci.lane.getLastStepHaltingNumber(lane) 
           MaxSpeed = traci.lane.getMaxSpeed(lane)
           rw[nblane*i+j] = lsvn[nblane*i+j]*(lsms[nblane*i+j]-MaxSpeed)/MaxSpeed
           if lsvn[nblane*i+j] > 0:
               lsmw[nblane*i+j] = lswt[nblane*i+j]**2/lsvn[nblane*i+j] # square of mean waiting time times number of vehicles = square of total waiting time over number of vehicles 
           
               # sometime it crashes on print(traci.vehicle.getNextTLS(traci.lane.getLastStepVehicleIDs(lane)[-1])[0]).
               # the result of traci.vehicle.getNextTLS(traci.lane.getLastStepVehicleIDs(lane)[-1])) was ()
               # it's unclear how I got no traffic light though. Potentially, the vehicle was considered having already pass the traffic light 
               #print(traci.vehicle.getNextTLS(traci.lane.getLastStepVehicleIDs(lane)[-1])) # (('21748279', 19, 5.100500237275224, 'G'),) # only one tl, so only one tupple
               tmp = traci.vehicle.getNextTLS(traci.lane.getLastStepVehicleIDs(lane)[-1])
               if len(tmp) > 0:
                   #print(traci.vehicle.getNextTLS(traci.lane.getLastStepVehicleIDs(lane)[-1])[0]) # ('21748279', 19, 5.100500237275224, 'G') # (tlsID, tlsIndex, distance, state)
                   #print(traci.vehicle.getNextTLS(traci.lane.getLastStepVehicleIDs(lane)[-1])[0][2]) # 5.100500237275224
                   dhead[nblane*i+j] = traci.vehicle.getNextTLS(traci.lane.getLastStepVehicleIDs(lane)[-1])[0][2]
    #print()
    # for i, edge in enumerate(OutLanesEdge):
        # for j, lane in enumerate(edge):
           # #print(i,j,lane)
           # lsvn[nblane*i+j+12] = traci.lane.getLastStepVehicleNumber(lane)
           # lsms[nblane*i+j+12] = traci.lane.getLastStepMeanSpeed(lane)
    
    rw1 = np.sum(rw)
    rw2 = -np.sum(lsmw)/1000 # I should also consider the waiting time of the vehicle
    #rw = rw1+rw2
    rw = np.array([rw1, rw2])
    #print('flow reward', rw1, 'waiting reward', rw2, 'total reward', rw)
    # np.sum(rw) ~ -78 to -150
    # -np.sum(lswt)/3 ~ -0 to -228 tend to slowly increase and then drop fast 
    obs = np.concatenate([lsvn,lsms,lswt,lshn,dhead],axis=0)
    return obs, rw
    # Useful information:
    # traci.lane.getLastStepVehicleNumber(lane)
    # traci.lane.getLastStepMeanSpeed(lane)
    # getLastStepVehicleIDs
    # getLastStepHaltingNumber # total number of halting vehicles. A speed of less than 0.1 m/s is considered a halt.
    # getWaitingTime # Returns the waiting time for all vehicles on the lane [s]
    # traci.vehicle.lane.getLastStepIDs(laneID)[-1] # head of the queue vehicle 
    # traci.vehicle.getWaitingTime(traci.vehicle.lane.getLastStepIDs(laneID)[-1]) # waiting time of head
    # traci.vehicle.getNextTLS(traci.vehicle.lane.getLastStepIDs(laneID)[-1])[0][2] # getNextTLS return [(tlsID, tlsIndex, distance, state), ...]
    # waiting time: time (in seconds) spent with a speed below 0.1m/s since the last time it was faster than 0.1m/s
    
def dqnlearn(num_timesteps, num_simulations, *args, **kwargs):
    yellowTime = 3 # I have Warning: Vehicle '3.527' performs emergency stop at the end of lane for 3
    redTime = 0
    itBetweenDecision = int((yellowTime+redTime+3)/stepLength) # each itteration is stepLength s, I need at least 3s of green time
    
    lightList = traci.trafficlight.getIDList()
    #obsLen = 2*4*3*2 # max of in and out lanes, 4 directions, 3 lanes per edges, 2 information per lane
    
    alg = []
    lanes = []
    lanesEdge = []
    OutLanes = []
    OutLanesEdge = []
    rewardsLength = 0
    for i, tl in enumerate(lightList):
        phases = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl)[0].getPhases()
        num_phases = len(phases)//2 # The tl have 4 or 6 phases in their base program, so so it can be 2 or 3
        ph = []
        for phase in phases[::2]: # skip yellow
            ph.append(phase._phaseDef)
        print('phase', ph)
            
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
        edgeList = [None]
        oldlane = None
        OutLanesEdge.append([])
        for lane in lanes[-1]: # 1st dim: tl, 2nd dim: one for each lane
            if lane == oldlane:
                continue
            oldlane = lane
            if traci.lane.getEdgeID(lane) == edgeList[-1]:
                lanesEdge[-1][-1].append(lane)
                #OutLanesEdge[-1].append(outlanedic[lane])
                #print(lane, outlanedic[lane])
            else:
                lanesEdge[-1].append([lane])
                #OutLanesEdge[-1].append([outlanedic[lane]])
                #print('new edge',lane, outlanedic[lane])
                edgeList.append(traci.lane.getEdgeID(lane))
        for iedge, edge in enumerate(edgeList[1:]): # get the 2 lanes that are before the junction where the road goes from 2 to 3 lanes (will need to be change for different road network)
            for ilane in range(2):
                lanesEdge[-1][iedge].append(edge[:-1]+'_'+str(ilane)) # I wrote that for this network. Not general            
                
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
        #return None # stop here to look at the lanes
        obs, rw = getobs(tl, num_phases,lanesEdge[-1]) #, OutLanesEdge[-1]) # first dim of lanesEdge is tl, 2nd dim is edge (direction), value is lane (3rd dim)
        alg.append(dqn.QLearner(num_phases = num_phases,observation_shape = len(obs),scope=str(i), phases=ph, timeBetweenDecision=itBetweenDecision*stepLength, *args, **kwargs))
        alg[i].last_obs = obs*itBetweenDecision # because we will divide by that afterwards
           
    # neighTL = [] # I would like to know for sure which one is left and which one is right. Maybe use lanesEdge
    # # Comparing lanes in common doesn't work, because only incoming lanes are in the list
    # for i, tl in enumerate(lightList):
        # neigh = np.array([-1,-1]) # only 2 neighbors in this map
        # for j, tl2 in enumerate(lightList):
            # if i != j:
# #                if i == 0 and j == 4:
# #                    print(lanes[i])
# #                    print(OutLanes[j])
# #                    print(lanes[i][:len(lanes[i])//2])
# #                    print(lanes[i][len(lanes[i])//2:])
# #                    print(set(lanes[i][:len(lanes[i])//2]).intersection(OutLanes[j]))
# #                    print(set(lanes[i][len(lanes[i])//2:]).intersection(OutLanes[j]))
# #                    print()
                # if len(set(lanes[i][:len(lanes[i])//2]).intersection(OutLanes[j]))>0:
                   # neigh[0] = j
                # elif len(set(lanes[i][len(lanes[i])//2:]).intersection(OutLanes[j]))>0:
                   # neigh[1] = j
        # #print('neighbor tl', i, neigh, tl)
        # neighTL.append(neigh) # 1st dim of neighTL is tl, value is the neighbor tl (2nd dim)
    # # 0,4,5,1,3,2 # 0 didn't get 4
    # neighTL[0][1] = 4 # correct manually. Due to edge expanding from 3 lanes to 4 lanes
    # # I will need a better way on another map
    
    saver = tf.train.Saver() # create saver
    #saveFrequency = 10000 
    saveFile = r"/model/A.ckpt"
    if False: # put True here if I want to restore
        saver.restore(kwargs['session'], saveFile)
        print("Model restored.")
        for i in range(len(alg)):
            alg[i].model_initialized = True # I believe it is initialized if it is restored
    
    episode_rewards = []
    action = np.empty_like(alg, int) #  Need to save all actions and then use them to learn each q
    reward = np.empty((len(alg),2), float) # now I have 2 rewards 
    # hid = np.empty((len(alg),kwargs['num_hidden']),float)
    # Hs = np.empty((len(alg),2*kwargs['num_hidden']),float) # 2 because each tl has 2 neighbors
    #it = 0
    for isim in range(num_simulations):
        for it in range(num_timesteps):
            if it == 1000: 
                return None 
            for q in alg:
                q.t += 1 # upgrade timestep here
            #print(it, traci.simulation.getTime())
            #if (it+1) % saveFrequency == 0: # save model # Don't save for now
            #    saver.save(kwargs['session'], saveFile)
            #    print("Model saved.")
            if traci.simulation.getMinExpectedNumber() <= 0:
                print('Error: No more cars')
                break
            if it%itBetweenDecision == 0:
                if it>0: #not first time
                    episode_rewards.append(reward/itBetweenDecision)
                done = it > num_timesteps - itBetweenDecision # done if it's the last action 
                for i, tl in enumerate(lightList):
                    if it>0: #not first time 
                        #info = None
                        #print('reward',reward[i], np.sum(reward[i]))
                        alg[i].replay_buffer.store_effect(idx, action[i], np.sum(reward[i]), done)
                        #alg[i].replay_buffer.store_effect(idx, action[i], reward[i], done, Hs[i])
                        alg[i].update_model()
                    
                    lastObs = alg[i].last_obs/itBetweenDecision
                    #idx = alg[i].replay_buffer.store_frame(lastObs, 2*kwargs['num_hidden'])
                    idx = alg[i].replay_buffer.store_frame(lastObs)
                    #lastObs = alg[i].replay_buffer.encode_recent_observation() # No need to encode, I'm not using context
            
                    if not alg[i].model_initialized:
                        alg[i].model_initialized = True
                        initialize_interdependent_variables(alg[i].session, tf.global_variables(), {
                                alg[i].obs_t_ph: lastObs[np.newaxis,:],
                                                      })
                    #hid[i] = alg[i].session.run(alg[i].hid, feed_dict={alg[i].obs_t_ph:lastObs[np.newaxis,:]})
                
                alg[0].log_progress(episode_rewards, it) # need to log only once.
                
                if done:
                    print('done')
                    continue # no need to finish if it's done. it will be less than itBetweenDecision iterations
              
            # Choose action (with epsilon greedy)
                for i, tl in enumerate(lightList):
                    # h = []
                    # for ntl in neighTL[i]:
                        # if ntl != -1:
                            # h.append(hid[ntl,:])
                        # else:
                            # h.append(np.zeros((kwargs['num_hidden'],)))
                    # h = np.concatenate(h,axis=0)
                    # Hs[i] = h
                    if random.random() > alg[i].exploration.value(alg[i].t):
                        #q = alg[i].session.run(alg[i].q, feed_dict={alg[i].obs_t_ph:lastObs[np.newaxis,:],alg[i].tl_ph:h[np.newaxis,:]})
                        q = alg[i].session.run(alg[i].q, feed_dict={alg[i].obs_t_ph:lastObs[np.newaxis,:]})
                        action[i] = np.argmax(q,axis=1)
                        #print('chosen action', q, action[i])
                    else:
                        action[i] = alg[i].randomAction() #random.randrange(alg[i].num_phases) # random integer
                        #print('random action', action[i], alg[i].num_phases)
                    alg[i].updatePhase(action[i])

                    current_phase = traci.trafficlight.getRedYellowGreenState(tl)
                    newphase = alg[i].phases[action[i]]
                    print(i, action[i], current_phase, newphase)
                    if newphase != current_phase: # not same phase
                        traci.trafficlight.setPhaseDuration(tl, 0) # trying to force switch by putting duration to 0. Actually, 0 made it swtich to red, so let's try yellowtime
                        #program = traci.trafficlight.Logic(programID, type, currentPhaseIndex, phases=None, subParameter=None)
                        yellowphase = list(current_phase)
                        #redphase = list(current_phase)
                        for ichar, char in enumerate(current_phase):
                            if newphase[ichar] == 'r' and char != 'r':
                                yellowphase[ichar] = 'y'
                                #redphase[ichar] = 'r'
                        yellowphase = ''.join(yellowphase)
                        #redphase = ''.join(redphase) # Says duration are in miliseconds, but that was in 2012 they might have changed it
                        phases = [traci.trafficlight.Phase(phaseDef=current_phase,duration=0,duration1=0,duration2=0),
                                  traci.trafficlight.Phase(phaseDef=yellowphase,duration=yellowTime,duration1=yellowTime,duration2=yellowTime),
                                  #traci.trafficlight.Phase(phaseDef=redphase,duration=redTime,duration1=redTime,duration2=redTime),
                                  traci.trafficlight.Phase(phaseDef=newphase,duration=10000,duration1=10000,duration2=10000)]
                        program = traci.trafficlight.Logic(phases=phases, subID='0', type=0, subParameter=0, currentPhaseIndex=0)
                        traci.trafficlight.setCompleteRedYellowGreenDefinition(tl, program)
                        #print(traci.trafficlight.getRedYellowGreenState(tl))
              
            print('Before step', it, traci.trafficlight.getRedYellowGreenState(tl))
            traci.simulationStep()
            print('After step', it, traci.trafficlight.getRedYellowGreenState(tl))
            for i, tl in enumerate(lightList):
                #print('    ', i, traci.trafficlight.getRedYellowGreenState(tl), traci.trafficlight.getNextSwitch(tl), traci.trafficlight.getPhaseDuration(tl))
                obs, rw = getobs(tl, alg[i].num_phases,lanesEdge[i]) #,OutLanesEdge[i])
                #print('obs', obs)
                #print('rw', rw)
                if it%itBetweenDecision == 0:
                    alg[i].last_obs = obs
                    reward[i] = rw
                else:
                    alg[i].last_obs += obs
                    reward[i] += rw
            
        traci.close()
        saver.save(kwargs['session'], saveFile)
        print("Model saved.")
        print('End of one simulation', 'iteration this time', len(episode_rewards)- rewardsLength, 'mean reward', np.mean(episode_rewards[rewardsLength:],axis=0))
        rewardsLength = len(episode_rewards)
        printMyRoute(num_timesteps, isim+1) # create new route with new probability. +1 because isim hasn't increased yet 
        
        traci.start([r"D:\InstalledProgram\SUMO\bin\sumo.exe", "-c", "conf2.sumocfg","--tripinfo-output", "tripinfo"+str(isim)+".xml","--no-step-log","--time-to-teleport", "-1"])

def main():
    num_timesteps = 5000
    num_simulations = 30 # I will restart the simulation periodically so it can learn from a fresh start
    num_iterations = num_timesteps*num_simulations
    #sumoBinary = r"D:\InstalledProgram\SUMO\bin\sumo-gui.exe"
    sumoBinary = r"D:\InstalledProgram\SUMO\bin\sumo.exe"
        
    # Run training
    # f = open('triedParameters.txt','a+')
    # ntries = 100
    # for ntry in range(ntries):
        # # hyperparameter random search
        # lr_multiplier = 10**random.uniform(-2, 2) #1.0
        # num_hidden = random.randint(30,100)
        # gamma = random.uniform(0.8, 0.95) # horizon should be 1/(1-gamma). We are getting reward every 6 seconds, so 0.9 for 60 seconds should be good
        # learning_freq = random.randint(1,20)
        # target_update_freq = random.randint(10,200)
        # explor1 = random.randint(100,1000)
        # explor2 = int(num_iterations*random.uniform(0.1,0.9))
        # if explor2 < explor1:
            # explor2 = (num_iterations+explor1)/2
        # print(ntry, lr_multiplier, num_hidden, gamma, learning_freq, target_update_freq, explor1, explor2, file=f)
    lr_multiplier = 1
    num_hidden = 100
    gamma = 0.9
    learning_freq = 10
    target_update_freq = 100
    explor1 = num_timesteps 
    explor2 = num_timesteps*(num_simulations-1)
    
    isim = 0
    printMyRoute(num_timesteps, isim)
    traci.start([sumoBinary, "-c", "conf2.sumocfg","--tripinfo-output", "tripinfo"+str(isim)+".xml","--no-step-log","--time-to-teleport", "-1"])
    seed = random.randint(0, 9999)
    print('random seed = %d' % seed)
    session = get_session()
    #rew_file='reward'+str(ntry)+'.pkl'
    rew_file='MultiSIMreward.pkl'

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
            (8*num_timesteps, 0.5),
            (15*num_timesteps, 0.1),
            (explor2, 0.01),
        ], outside_value=0. # No exploration towards the end so we can see the true reward
    )

    #dqnlearn(
    dqnlearn(num_timesteps=num_timesteps,
        num_hidden = num_hidden,
        optimizer_spec=optimizer,
        session=session,
        num_simulations = 20,
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
