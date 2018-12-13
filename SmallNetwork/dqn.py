import uuid
import time
import pickle
import sys
#import gym.spaces
#import itertools
import numpy as np
import random
import random
import tensorflow                as tf
#import tensorflow.contrib.layers as layers
from collections import namedtuple
from dqn_utils import *
import tensorflow.contrib.layers as layers

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

class QLearner(object):
    
  def firstLayer(self, inp, num_outputs, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inp
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=tf.nn.relu)
        return out
  
  def middle(self, inp, num_outputs, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inp
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=tf.nn.relu)
        return out
    
  def lastLayer(self, inp, num_phases, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inp
        out = layers.fully_connected(out, num_outputs=num_phases, activation_fn=None)
        return out

  def __init__(
    self,
    observation_shape, # new
    num_phases,
    num_hidden,
    scope,
    phases,
    timeBetweenDecision,
    #q_func,
    optimizer_spec,
    session,
    exploration=LinearSchedule(1000000, 0.1),
    replay_buffer_size=1000000,
    batch_size=32,
    gamma=0.99,
    learning_starts=50000,
    learning_freq=4,
    frame_history_len=4,
    target_update_freq=10000,
    grad_norm_clipping=10,
    rew_file='reward.pkl',
    double_q=True):
    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            img_in: tf.Tensor
                tensorflow tensor representing the input image
            num_actions: int
                number of actions
            scope: str
                scope in which all the model related variables
                should be created
            reuse: bool
                whether previously created variables should be reused.
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    session: tf.Session
        tensorflow session to use.
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    stopping_criterion: (env, t) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    grad_norm_clipping: float or None
        If not None gradients' norms are clipped to this value.
    double_q: bool
        If True, then use double Q-learning to compute target values. Otherwise, use vanilla DQN.
        https://papers.nips.cc/paper/3964-double-q-learning.pdf
    """

    self.target_update_freq = target_update_freq
    self.optimizer_spec = optimizer_spec
    self.batch_size = batch_size
    self.learning_freq = learning_freq
    self.learning_starts = learning_starts
    self.session = session
    self.exploration = exploration
    self.rew_file = str(uuid.uuid4()) + '.pkl' if rew_file is None else rew_file
    
    self.num_phases = num_phases
    self.phases = phases
    self.currentPhase = 0
    self.timeBetweenDecision = timeBetweenDecision
    self.timeCurrentPhase = 0
    self.Iter = 0 # the actual decision while self.t is each iteration in the simulator 

    ###############
    # BUILD MODEL #
    ###############
    # set up placeholders
    # placeholder for current observation (or state)
    self.obs_t_ph              = tf.placeholder(tf.float32, [None,observation_shape])
    # placeholder for current action
    self.act_t_ph              = tf.placeholder(tf.int32,   [None])
    # placeholder for current reward
    self.rew_t_ph              = tf.placeholder(tf.float32, [None])
    # placeholder for next observation (or state)
    self.obs_tp1_ph            = tf.placeholder(tf.float32, [None,observation_shape])
    # placeholder for end of episode mask
    self.done_mask_ph          = tf.placeholder(tf.float32, [None])
    # placeholder for hidden state of other tl
    #self.tl_ph              = tf.placeholder(tf.float32, [None,2*num_hidden])
    
    #print('scope',scope)
    #print('observation_shape', observation_shape, 'num_hidden', num_hidden, 'num_phases', num_phases)
    self.hid = self.firstLayer(self.obs_t_ph, num_hidden, scope="hid"+scope, reuse=tf.AUTO_REUSE)
    #print('hid', self.hid)
    #hid = tf.concat([self.hid,self.tl_ph],axis=1)
    hid = self.hid
    #print('concat', hid)
    self.mid = self.middle(hid, 2*num_hidden, scope="mid"+scope, reuse=tf.AUTO_REUSE)
    #print('mid', self.mid)
    self.q = self.lastLayer(self.mid, num_phases, scope="q_func"+scope, reuse=tf.AUTO_REUSE) # change scope each time so it's not reused
    #print('qfunc'+scope, self.q)
    self.target_hid = self.firstLayer(self.obs_tp1_ph, num_hidden, scope="target_hid"+scope, reuse=tf.AUTO_REUSE) # Important target compute next state Q-value 
    #print('target_hid', self.target_hid)
    #target_hid = tf.concat([self.target_hid,self.tl_ph],axis=1)
    target_hid = self.target_hid
    #print('target_concat', target_hid)
    self.target_mid = self.middle(target_hid, 2*num_hidden, scope="target_mid"+scope, reuse=tf.AUTO_REUSE)
    #print('target_mid', self.target_mid)
    self.target_q = self.lastLayer(self.target_mid, num_phases, scope="target_q_func"+scope, reuse=tf.AUTO_REUSE) # change scope each time so it's not reused
    #print("target_q_func"+scope, self.target_q)

    #self.q = q_func(self.obs_t_ph, num_phases, scope="q_func", reuse=False)
    #self.target_q = q_func(self.obs_tp1_ph, num_phases, scope="target_q_func", reuse=False)
    qa = tf.reduce_sum(self.q*tf.one_hot(self.act_t_ph,self.num_phases),axis=1)
    #print('qa', qa)
    if double_q:
        target = tf.reduce_sum(self.target_q*tf.one_hot(tf.argmax(self.q,axis=1),self.num_phases),axis=1)
        Bellman_error = qa - self.rew_t_ph - gamma*tf.stop_gradient(target)
    else:
        Bellman_error = qa - self.rew_t_ph - gamma*tf.stop_gradient(tf.reduce_max(self.target_q, axis=1))
    #print(Bellman_error)
    self.total_error = tf.reduce_mean(huber_loss(Bellman_error))
    #print(self.total_error)
    #print()
    
    # must not include adam variable, must not unclude previous scope (so q_funct must only be q_func+scope)
    #print('target_(hid|mid|q_func{})'.format(scope))
    # negative lookahead (?!...) for Adam (?!Adam)
    #print('(hid|mid|q_func{})(?!.*Adam)'.format(scope))
    #q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=r'(hid|mid|q_func).*')
    #target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=r'target_(hid|mid|q_func).*')
    q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='(hid|mid|q_func{})(?!.*Adam)'.format(scope))
    target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_(hid|mid|q_func{})'.format(scope))

    # construct optimization op (with gradient clipping)
    #print('a')
    self.learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
    #print('b')
    optimizer = self.optimizer_spec.constructor(learning_rate=self.learning_rate, **self.optimizer_spec.kwargs)
    #print('c')
    self.train_fn = minimize_and_clip(optimizer, self.total_error,
                 var_list=q_func_vars, clip_val=grad_norm_clipping)
    #print('d')

    # update_target_fn will be called periodically to copy Q network to target Q network
    update_target_fn = []
#    print()
#    print('q_func_vars',q_func_vars)
#    print()
#    print('target_q_func_vars',target_q_func_vars)
#    print()
#    print('q_func_vars',sorted(q_func_vars,        key=lambda v: v.name))
#    print()
#    print('target_q_func_vars',sorted(target_q_func_vars, key=lambda v: v.name))
#    print()
    
    for var, var_target in zip(sorted(q_func_vars,        key=lambda v: v.name),
                               sorted(target_q_func_vars, key=lambda v: v.name)):
        #print('var', var)
        #print('var_target', var_target)
        update_target_fn.append(var_target.assign(var))
    #print('e')
    self.update_target_fn = tf.group(*update_target_fn)
    #print('f')

    # construct the replay buffer
    self.replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)
    #print('g')
    self.replay_buffer_idx = None

    ###############
    # RUN ENV     #
    ###############
    self.model_initialized = False
    self.num_param_updates = 0
    self.mean_episode_reward      = -float('nan')
    self.best_mean_episode_reward = -float('inf')
    self.log_every_n_steps = 1000 # 10000

    self.start_time = None
    self.t = 0

  def step_env(self):
    ### 2. Step the env and store the transition
    # At this point, "self.last_obs" contains the latest observation that was
    # recorded from the simulator. Here, your code needs to store this
    # observation and its outcome (reward, next observation, etc.) into
    # the replay buffer while stepping the simulator forward one step.
    # At the end of this block of code, the simulator should have been
    # advanced one step, and the replay buffer should contain one more
    # transition.
    # Specifically, self.last_obs must point to the new latest observation.
    # Useful functions you'll need to call:
    # obs, reward, done, info = env.step(action)
    # this steps the environment forward one step
    # obs = env.reset()
    # this resets the environment if you reached an episode boundary.
    # Don't forget to call env.reset() to get a new observation if done
    # is true!!
    # Note that you cannot use "self.last_obs" directly as input
    # into your network, since it needs to be processed to include context
    # from previous frames. You should check out the replay buffer
    # implementation in dqn_utils.py to see what functionality the replay
    # buffer exposes. The replay buffer has a function called
    # encode_recent_observation that will take the latest observation
    # that you pushed into the buffer and compute the corresponding
    # input that should be given to a Q network by appending some
    # previous frames.
    # Don't forget to include epsilon greedy exploration!
    # And remember that the first time you enter this loop, the model
    # may not yet have been initialized (but of course, the first step
    # might as well be random, since you haven't trained your net...)

    #####
    
    idx = self.replay_buffer.store_frame(self.last_obs)
    lastObs = self.replay_buffer.encode_recent_observation()

    if not self.model_initialized:
        self.model_initialized = True
        initialize_interdependent_variables(self.session, tf.global_variables(), {
          self.obs_t_ph: lastObs[np.newaxis,:],
        })
    
    # Choose action (with epsilon greedy)
    # TO DO LATER: find new q for each trafic light (num_actions)
    if random.random() > self.exploration.value(self.t):
        q = self.session.run(self.q, feed_dict={self.obs_t_ph:lastObs[np.newaxis,:]})
        action = np.argmax(q,axis=1)
    else:
        action = random.randrange(self.num_phases)
    return action, idx

  def update_model(self):
    ### 3. Perform experience replay and train the network.
    # note that this is only done if the replay buffer contains enough samples
    # for us to learn something useful -- until then, the model will not be
    # initialized and random actions should be taken
    #print('update_model', self.t, self.learning_starts, self.learning_freq, self.replay_buffer.can_sample(self.batch_size))
    if (self.t > self.learning_starts and \
        self.Iter % self.learning_freq == 0 and \
        self.replay_buffer.can_sample(self.batch_size)):
      # Here, you should perform training. Training consists of four steps:
      # 3.a: use the replay buffer to sample a batch of transitions (see the
      # replay buffer code for function definition, each batch that you sample
      # should consist of current observations, current actions, rewards,
      # next observations, and done indicator).
      # 3.b: initialize the model if it has not been initialized yet; to do
      # that, call
      #    initialize_interdependent_variables(self.session, tf.global_variables(), {
      #        self.obs_t_ph: obs_t_batch,
      #        self.obs_tp1_ph: obs_tp1_batch,
      #    })
      # where obs_t_batch and obs_tp1_batch are the batches of observations at
      # the current and next time step. The boolean variable model_initialized
      # indicates whether or not the model has been initialized.
      # Remember that you have to update the target network too (see 3.d)!
      # 3.c: train the model. To do this, you'll need to use the self.train_fn and
      # self.total_error ops that were created earlier: self.total_error is what you
      # created to compute the total Bellman error in a batch, and self.train_fn
      # will actually perform a gradient step and update the network parameters
      # to reduce total_error. When calling self.session.run on these you'll need to
      # populate the following placeholders:
      # self.obs_t_ph
      # self.act_t_ph
      # self.rew_t_ph
      # self.obs_tp1_ph
      # self.done_mask_ph
      # (this is needed for computing self.total_error)
      # self.learning_rate -- you can get this from self.optimizer_spec.lr_schedule.value(t)
      # (this is needed by the optimizer to choose the learning rate)
      # 3.d: periodically update the target network by calling
      # self.session.run(self.update_target_fn)
      # you should update every target_update_freq steps, and you may find the
      # variable self.num_param_updates useful for this (it was initialized to 0)
      #####

      # YOUR CODE HERE
      #obs_batch, act_batch, rew_batch, next_obs_batch, done_mask, hid_batch = self.replay_buffer.sample(self.batch_size)
      obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = self.replay_buffer.sample(self.batch_size)
      if not self.model_initialized:
          self.model_initialized = True
          initialize_interdependent_variables(self.session, tf.global_variables(), {
              self.obs_t_ph: obs_batch,
              self.obs_tp1_ph: next_obs_batch
          })
    
#      print('obs_batch',self.obs_t_ph)
#      print(obs_batch.shape)
#      print('act_batch',self.act_t_ph)
#      print(act_batch.shape)
#      print('rew_batch',self.rew_t_ph)
#      print(rew_batch.shape)
#      print('next_obs_batch',self.obs_tp1_ph)
#      print(next_obs_batch.shape)
#      print('done_mask',self.done_mask_ph)
#      print(done_mask.shape)
#      print('learning_rate',self.learning_rate)
#      print(self.optimizer_spec.lr_schedule.value(self.t))
      
      # Need to also have hidden state
      # self.tl_ph
      
      #self.session.run([self.train_fn,self.total_error], feed_dict={self.obs_t_ph:obs_batch,self.act_t_ph:act_batch,self.rew_t_ph:rew_batch,self.obs_tp1_ph:next_obs_batch,self.done_mask_ph:done_mask,self.learning_rate:self.optimizer_spec.lr_schedule.value(self.t)
      #,self.tl_ph:hid_batch})
      _, total_error = self.session.run([self.train_fn,self.total_error], feed_dict={self.obs_t_ph:obs_batch,self.act_t_ph:act_batch,self.rew_t_ph:rew_batch,self.obs_tp1_ph:next_obs_batch,self.done_mask_ph:done_mask,self.learning_rate:self.optimizer_spec.lr_schedule.value(self.t)
      })
      #print()
      #print('total_error', total_error)
      #print()
    
      if self.num_param_updates % self.target_update_freq == 0:
            self.session.run(self.update_target_fn)      
    
      #############################
      self.num_param_updates += 1

    self.Iter += 1

  def log_progress(self, episode_rewards, it):
    if len(episode_rewards) > 0:
      self.mean_episode_reward = np.mean(episode_rewards[-100:])

    # I add that
    #self.ListTimestep = 
    #print(len(episode_rewards), self.t, self.t % self.log_every_n_steps, self.model_initialized, self.rew_file, self.log_every_n_steps)
    if len(episode_rewards) > 100:
      self.best_mean_episode_reward = max(self.best_mean_episode_reward, self.mean_episode_reward)

    #print('log_progress', self.t, self.log_every_n_steps, self.t % self.log_every_n_steps)
    if it % self.log_every_n_steps == 0 and self.model_initialized:
      print("Timestep", it, self.t)
      print("mean reward (100 episodes) %f" % self.mean_episode_reward)
      print("best mean reward %f" % self.best_mean_episode_reward)
      print("episodes %d" % len(episode_rewards))
      print("exploration %f" % self.exploration.value(self.t))
      print("learning_rate %f" % self.optimizer_spec.lr_schedule.value(self.t))
      if self.start_time is not None:
        print("running time %f" % ((time.time() - self.start_time) / 60.))

      self.start_time = time.time()

      sys.stdout.flush()

      with open(self.rew_file, 'wb') as f:
        pickle.dump(episode_rewards, f, pickle.HIGHEST_PROTOCOL)
        
      #with open(self.rew_file, 'wb') as f:
      #  pickle.dump([self.t, self.mean_episode_reward, self.best_mean_episode_reward, episode_rewards], f, pickle.HIGHEST_PROTOCOL)

#def learn(*args, **kwargs):
#  alg = QLearner(*args, **kwargs)
#  while not alg.stopping_criterion_met():
#    alg.step_env()
    # at this point, the environment should have been advanced one step (and
    # reset if done was true), and self.last_obs should point to the new latest
    # observation
#    alg.update_model()
#    alg.log_progress()

  def updatePhase(self, action):
      self.currentPhase = action
      self.timeCurrentPhase = self.t
      
  def randomAction(self):
    if self.currentPhase % 2 == 0: # even should last a long time
        if 5*random.random() > self.t - self.timeCurrentPhase: # as self.t gets bigger 
            return self.currentPhase
    return random.randrange(self.num_phases)
  
      # if self.num_phases == 2:
          # if self.currentPhase == 0:
              # if random.random() > self.timeBetweenDecision/45:
                  # return 0
              # else:
                  # return 1
          # else:
              # if random.random() > self.timeBetweenDecision/45:
                  # return 1
              # else:
                  # return 0
      # else: # 4 phases
          # if self.currentPhase == 0:
              # if random.random() > self.timeBetweenDecision/27:
                  # return 0
              # else:
                  # return 1
          # elif self.currentPhase == 1:
              # if random.random() > self.timeBetweenDecision/6:
                  # return 1
              # else:
                  # return 2
          # elif self.currentPhase == 2:
              # if random.random() > self.timeBetweenDecision/27:
                  # return 2
              # else:
                  # return 3
          # else:
              # if random.random() > self.timeBetweenDecision/6:
                  # return 3
              # else:
                  # return 0
