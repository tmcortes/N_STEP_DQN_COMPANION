import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from statistics import mean
import pandas as pd
import os
import glob
import time
import collections
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import pickle
from DQN import DQN, ReplayMemory, Transition, init_weights, PrioritizedReplayMemory
from config import print_setup
import config as cfg
from aux import *
import argparse
import pdb
from datetime import datetime
import copy 
from collections import Counter
import warnings
from scipy.special import softmax


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', action='store_true', default=False, help="(bool) Inizializate the model with a pretrained model.")
parser.add_argument('--freeze', type=str, default='False', help="(bool) Inizializate the model with a pretrained moddel freezing the layers but the last one.")
parser.add_argument('--experiment_name', type=str, default=cfg.EXPERIMENT_NAME, help="(str) Name of the experiment. Used to name the folder where the model is saved. For example: my_first_DQN.")
parser.add_argument('--load_model', action='store_true', help="Load a checkpoint from the EXPERIMENT_NAME folder. If no episode is specified (LOAD_EPISODE), it loads the latest created file.")
parser.add_argument('--load_episode', type=int, default=0, help="(int) Number of episode to load from the EXPERIMENT_NAME folder, as the sufix added to the checkpoints when the save files are created. For example: 500, which will load 'model_500.pt'.")
parser.add_argument('--batch_size', type=int, default=cfg.BATCH_SIZE, help="(int) Batch size for the training of the network. For example: 54.")
parser.add_argument('--lr', type=float, default=cfg.LR, help="(float) Learning rate. For example: 1e-3.")
parser.add_argument('--replay_memory', type=int, default=cfg.REPLAY_MEMORY, help="(int) Size of the Experience Replay memory. For example: 1000.")
parser.add_argument('--gamma', type=float, default=cfg.GAMMA, help="(float) Discount rate of future rewards. For example: 0.99.")
parser.add_argument('--eps_start', type=float, default=cfg.EPS_START, help="(float) Initial exploration rate. For example: 0.99.")
parser.add_argument('--eps_end', type=float, default=cfg.EPS_END, help="(float) Terminal exploration rate. For example: 0.05.")
parser.add_argument('--eps_decay', type=int, default=cfg.EPS_DECAY, help="(int) Decay factor of the exploration rate. Episode where the epsilon has decay to 0.367 of the initial value. For example: num_episodes/2.")
parser.add_argument('--energy_penalty', type=float, default=cfg.FACTOR_ENERGY_PENALTY, help="(int) Energy penalty factor, weight of doing unnecesary actions.")
parser.add_argument('--speed', type=float, default=cfg.SPEED_ROBOT, help="(float) Factor that multiplies the speed of the robot. The baseline speed its the avergae of a healthy person.")
parser.add_argument('--failure_rate', type=float, default=cfg.ERROR_PROB, help="(float) Factor that controls the failure rate of the robot, 0-1 (1 always fails)")
parser.add_argument('--n_step', type=int, default=cfg.N_STEP, help="(int) Number of time steps of the past used to make predictions")
parser.add_argument('--target_update', type=int, default=cfg.TARGET_UPDATE, help="(int) Frequency of the update of the Target Network, in number of episodes. For example: 10.")
parser.add_argument('--root', type=str, default=cfg.ROOT, help="(str) Name of the root folder for the saving of checkpoints. Parent folder of EXPERIMENT_NAME folders. For example: ./Checkpoints/")
parser.add_argument('--display', action='store_true', default=False, help="Display environment info as [Current state, action taken, transitioned state, immediate reward, total reward].")
parser.add_argument('--cuda', action='store_true', default=True, help="Use GPU if available.")
args = parser.parse_args()

# CONFIGURATION PARAMETERS
PRETRAINED = args.pretrained
FREEZE = args.freeze
EXPERIMENT_NAME = args.experiment_name
LOAD_MODEL = args.load_model
LOAD_EPISODE = args.load_episode
REPLAY_MEMORY = args.replay_memory
BATCH_SIZE = args.batch_size
GAMMA = args.gamma
EPS_START = args.eps_start
EPS_END = args.eps_end
EPS_DECAY = args.eps_decay
LR = args.lr
TARGET_UPDATE = args.target_update
SPEED_ROBOT = args.speed
FACTOR_ENERGY_PENALTY = args.energy_penalty
ROBOT_ACTION_DURATIONS = copy.deepcopy(cfg.ROBOT_AVERAGE_DURATIONS)
ROBOT_ACTION_DURATIONS.update((x, y*SPEED_ROBOT) for x, y in ROBOT_ACTION_DURATIONS.items())
ERROR_PROB = args.failure_rate
TAU = cfg.TAU
POSITIVE_REWARD = cfg.POSITIVE_REWARD
NO_ACTION_PROBABILITY = cfg.NO_ACTION_PROBABILITY
ROBOT_CONT_ACTIONS_MEANINGS = cfg.ROBOT_CONT_ACTIONS_MEANINGS
DECISION_RATE = cfg.DECISION_RATE
ONLY_RECOGNITION = cfg.ONLY_RECOGNITION
NORMALIZE_DATA = cfg.NORMALIZE_DATA
PERSONALIZATION = cfg.PERSONALIZATION
TYPE_PERSONALIZATION = cfg.TYPE_PERSONALIZATION
ROOT = args.root
N_STEP = args.n_step
PROB_SAVE = cfg.PROB_SAVE
HEALTHY = False
array_conf = [FACTOR_ENERGY_PENALTY,ROBOT_ACTION_DURATIONS, ERROR_PROB,HEALTHY]
IMPORTANCE_SAMPLING= cfg.IMPORTANCE_SAMPLING
TEMPORAL_CTX = cfg.TEMPORAL_CONTEXT
device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

# --------------------------------
#Lists to debug training
total_loss = [] #List to save the mean values of the episode losses.
episode_loss = [] #List to save every loss value during a single episode.
total_reward = [] #List to save the total reward gathered each episode.
ex_rate = [] #List to save the epsilon value after each episode.

#Environment - Custom basic environment for kitchen recipes
env = gym.make("gym_basic:basic-v0", display=args.display, disable_env_checker=True)

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H:%M")

# CREATION OF THE FOLDER WHERE THE RESULTS WILL BE SAVED
if PRETRAINED == True:
    pre = '_Using_pretained_model'
else:
    pre = ''
    
if FREEZE == True:
    freeze = '_Freezing_layers'
else: 
    freeze = ''
if NO_ACTION_PROBABILITY == 0:
    weight_prob = ''
else:
    weight_prob = '_NO_ACT_PROB_' + str(NO_ACTION_PROBABILITY)
    
if cfg.DECISION_RATE == 'random':
    decision_rate_name = '_DECISION_RATE_random_'
else:
    decision_rate_name = '_DECISION_RATE_'+str(DECISION_RATE)
    
if REPLAY_MEMORY > BATCH_SIZE:
    batch_name = '_CHANGING_BATCH_SIZE_MEMORY_'
else: 
    batch_name = ''
    
if cfg.Z_hidden_state == False:
    z_name = '_WITHOUT_Z_'
else: 
    z_name = '_WITH_Z_'
    
if FACTOR_ENERGY_PENALTY!=1:
    exp = '_EXP_ENERGY'
elif SPEED_ROBOT!= 1:
    exp = '_EXP_SPEED'
elif ERROR_PROB != 0:
    exp= '_EXP_FAILURE'
elif PERSONALIZATION == True:
    if TYPE_PERSONALIZATION == '':
        exp= 'NO_PERSONALIZATION_'+TYPE_PERSONALIZATION+ '_EPS_START_'+str(EPS_START)
    else:
        exp= 'PERSONALIZATION_'+TYPE_PERSONALIZATION+ '_EPS_START_'+str(EPS_START)
else:
    exp = '_BASELINE'
    

if ONLY_RECOGNITION:
    type_state = '_ONLY_RECOG_'
else:
    type_state = '_RECOG_AND_PRED_'
    
if NORMALIZE_DATA:
    normalize = 'NORM'
else:
    normalize = ''
if IMPORTANCE_SAMPLING:
    imp_samp_name = '_IMP_SAMP'
else:
    imp_samp_name = ''
if TEMPORAL_CTX:
    tmp_ctx = '_TMP_CTX'
else:
    tmp_ctx = ''

path = os.path.join(ROOT, EXPERIMENT_NAME + '_'+str(N_STEP)+'_STEPS_' + dt_string  +'_'+exp+tmp_ctx+imp_samp_name+'_REPLAY_MEM_'+str(REPLAY_MEMORY)+'_TARGET_UPDT_'+str(TARGET_UPDATE)+'LR_'+str(LR)+ '_GAMMA_'+str(GAMMA)+weight_prob+'_BATCH_'+str(BATCH_SIZE)+'_ENERGY_PENALTY_'+ str(FACTOR_ENERGY_PENALTY)+'_SPEED_'+str(SPEED_ROBOT)+'_ERROR_PROB_'+str(ERROR_PROB))

save_path = os.path.join(path, "Graphics") 
save_path_hist = os.path.join(save_path, "Histograms") 
save_path_memory = os.path.join(save_path, "Memory")
 
if not os.path.exists(path): os.makedirs(path)
if not os.path.exists(save_path): os.makedirs(save_path)
if not os.path.exists(save_path_hist): os.makedirs(save_path_hist)
if not os.path.exists(save_path_memory): os.makedirs(save_path_memory)


if ONLY_RECOGNITION:
    dataset = 'dataset_pred'
else:
    dataset = 'dataset_pred_recog_tmp_ctx'
    
root_realData = "./video_annotations/"+dataset+"/*" #!
videos_realData = glob.glob(root_realData) #Folders

if PERSONALIZATION: 
        if TYPE_PERSONALIZATION == '':
            with open("./video_annotations/"+cfg.PERSON+"_test.txt") as f:
                lines_x = f.readlines()
                lines = []
                for line in lines_x:
                    lines.append(root_realData.split('*')[0]+line.split('\n')[0])
            videos_realData_test=lines
            train_videos = list(set(videos_realData)-set(videos_realData_test))
            videos_realData = train_videos
        else:
            with open("./video_annotations/"+cfg.PERSON+"_train_finetuning_"+cfg.TYPE_PERSONALIZATION+".txt") as f:
                lines_x = f.readlines()
                lines = []
                for line in lines_x:
                    lines.append(root_realData.split('*')[0]+line.split('\n')[0])
            train_videos = lines
            videos_realData = train_videos
        
else: 
    with open("./video_annotations/"+cfg.TEST_FOLD+"_test.txt") as f:
        lines_x = f.readlines()
        lines = []
        for line in lines_x:
            lines.append(root_realData.split('*')[0]+line.split('\n')[0])

    videos_realData_test=lines
    train_videos = list(set(videos_realData)-set(videos_realData_test))
    videos_realData = train_videos


NUM_EPISODES = len(videos_realData)
print("Num episodes: ",NUM_EPISODES)
NUM_EPOCH = cfg.NUM_EPOCH

env.reset() #Set initial state
n_states = env.observation_space.n #Dimensionality of the input of the DQN
n_actions = env.action_space.n #Dimensionality of the output of the DQN 
hidden_size_LSTM = cfg.hidden_size_LSTM
#Networks and optimizer
policy_net = DQN(n_states, hidden_size_LSTM, n_actions).to(device)
target_net = DQN(n_states, hidden_size_LSTM, n_actions).to(device)


if PERSONALIZATION and TYPE_PERSONALIZATION != '':
    path_model = './Pretrained/person1.pt'
    checkpoint = torch.load(path_model)
    policy_net.load_state_dict(checkpoint['model_state_dict'])
    target_net.load_state_dict(checkpoint['model_state_dict'])
    EPS_START = 0.5
if PRETRAINED:
    path_model = './Pretrained/model_without_Z.pt'
    print("Without Z variable")
    print("\nUSING PRETRAINED MODEL---------------")
    policy_net.load_state_dict(torch.load(path_model))
    policy_net.to(device)
    EPS_START = 0.5
else:
    policy_net.apply(init_weights) # si no hay pretained


# ----------------------------------
optimizer = optim.AdamW(policy_net.parameters(), lr=LR,weight_decay=LR/10, amsgrad=True) 
# optimizer = optim.RMSprop(policy_net.parameters(), lr=LR,weight_decay=LR/10,centered=True)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = NUM_EPOCH,eta_min= LR/10)

if LOAD_MODEL:  
   
    path = os.path.join(ROOT, EXPERIMENT_NAME)
    if LOAD_EPISODE: 
        model_name = 'model_' + str(LOAD_EPISODE) + '.pt' #If an episode is specified
        full_path = os.path.join(path, model_name)

    else:
       
        list_of_files = glob.glob(path+ '/*') 
        full_path = max(list_of_files, key=os.path.getctime) #Get the latest file in directory

    print("-"*30)
    print("\nLoading model from ", full_path)
    checkpoint = torch.load(full_path)
    policy_net.load_state_dict(checkpoint['model_state_dict'])
    target_net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    LOAD_EPISODE = checkpoint['epoch']
    total_loss = checkpoint['loss']
    steps_done = checkpoint['steps']
    print("-"*30)




def softmax(x):
    return(np.exp(x)/np.exp(x).sum())

def circular_buffer():
    """
    circular buffer, will store temporary data of the actions (including idle state) 
    taken by the robot, one will be created for each video, i.e. each episode.
    """
    buffer = collections.deque(maxlen=N_STEP)
    return buffer 

def importance_sampling (state, hidden, action):
    
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY) 
    posible_actions = env.possible_actions_taken_robot()
    index_posible_actions = [i for i, x in enumerate(posible_actions) if x == 1]
    
    out, hidden = policy_net(state, hidden)
    n_out_b = out.detach().cpu().numpy()[0]
    n_out_filtered_b = n_out_b[index_posible_actions]
    out_action_b = n_out_b[action]
    new_index_action_b = np.where(n_out_filtered_b==out_action_b)[0][0]
    q_behavioral = softmax(n_out_filtered_b)[new_index_action_b]
    
    out, hidden = target_net(state, hidden)
    n_out_t = out.detach().cpu().numpy()[0]
    n_out_filtered_t = n_out_t[index_posible_actions]
    out_action_t = n_out_t[action]
    new_index_action_t = np.where(n_out_filtered_t==out_action_t)[0][0]
    q_target = softmax(n_out_filtered_t)[new_index_action_t]
    
    
    ##### FILTRAR POR POSIBLES ACCIONES ANTES DE HACER EL SOFTMAX

    imp_sampling = q_target/((1-eps_threshold)*q_behavioral + eps_threshold/len(index_posible_actions))
    
    # if imp_sampling>1:
    #     print("IMPORTANCE SAMPLING MAYOR QUE 1")
    #     pdb.set_trace()
    
    return torch.tensor([[imp_sampling]], device=device, dtype=torch.float32) 
    
def post_processed_possible_actions(out,index_posible_actions):
    """
    Function that performs a post-processing of the neural network output. 
    In case the output is an action that is not available, either because 
    of the object missing or left on the table, the most likely possible action will be selected 
    from the output of the neural network,
    Parameters
    ----------
    out : (tensor)
        DQN output.
    index_posible_actions : (list)
        Posible actions taken by the robot according to the objects available.
    Returns
    -------
    (tensor)
        Action to be performed by the robot.
    """
    action_pre_processed = out.max(1)[1].view(1,1)
     
    if action_pre_processed.item() in index_posible_actions:
        return action_pre_processed
    else:
        out = out.cpu().numpy()
        out = out[0]
   
        idx = np.argmax(out[index_posible_actions])
        action = index_posible_actions[idx]

        return torch.tensor([[action]], device=device, dtype=torch.long)
    

#Action taking
def select_action(state, hidden_train,hidden_val, phase):
    """
    Function that chooses which action to take in a given state based on the exploration-exploitation paradigm.
    This action will always be what is referred to as a possible action; the actions possible by the robot are 
    defined by the objects available in its environment.
    Input:
        state: (tensor) current state of the environment.
    Output:
        action: (tensor) either the greedy action (argmax Q value) from the policy network output, or a random action. 
    """
    global steps_done, i_epoch
    # print(steps_done)
    # print(i_epoch)
    sample = random.random() #Generate random number [0, 1]
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY) #Get current exploration rate
    posible_actions = env.possible_actions_taken_robot()
    index_posible_actions = [i for i, x in enumerate(posible_actions) if x == 1]

    if phase == 'val':
        with torch.no_grad():
            
            out, hidden_val = policy_net(state, hidden_val)
            
            if True in torch.isnan(out):
                pdb.set_trace()
            action = post_processed_possible_actions(out,index_posible_actions)
            # print(out)
            # print('action pre: ', action)
            # pdb.set_trace()
            return action, hidden_val
    else:
         if sample > eps_threshold and i_epoch > -1: #If the random number is higher than the current exploration rate, the policy network determines the best action.
             with torch.no_grad():
                 
                 out, hidden_train = policy_net(state, hidden_train)
                 # print(out)
                 if True in torch.isnan(out):
                     pdb.set_trace()
                 action = post_processed_possible_actions(out,index_posible_actions)
                 return action, hidden_train
         else:
             if NO_ACTION_PROBABILITY != 0:
                 index_no_action = index_posible_actions.index(5)
                 weights = [10]*len(index_posible_actions)
                 weights[index_no_action] = cfg.NO_ACTION_PROBABILITY
                 action = random.choices(index_posible_actions, weights, k=1)[0]
             else:
                 index_action = random.randrange(len(index_posible_actions))
                 action = index_posible_actions[index_action]
                 # print('posible actions: ',index_posible_actions)
                 # print('random action taken: ',action)
             return torch.tensor([[action]], device=device, dtype=torch.long), hidden_train


def action_rate(decision_cont,state,hidden_train,hidden_val, phase,prev_decision_rate):
    """
    Function that sets the rate at which the robot makes decisions.
    """
    if phase == 'train':
        hidden = hidden_train
    else:
        hidden = hidden_val
    if cfg.DECISION_RATE == "random":
        
        if phase == 'train':
            if decision_cont == 1:
                decision_rate = random.randrange(10,150)
                prev_decision_rate = decision_rate
            else:
                 decision_rate = prev_decision_rate
        else:
            decision_rate = 100
             
    else:
        decision_rate = cfg.DECISION_RATE
        prev_decision_rate = " "
        
    if decision_cont % decision_rate == 0:  
        action_selected, hidden = select_action(state, hidden_train,hidden_val, phase)
        flag_decision = True 
    else:
        action_selected = 5
        flag_decision = False
    # print("RANDOM NUMBER: ",decision_rate)
    # pdb.set_trace()
    return action_selected, flag_decision, prev_decision_rate, hidden
    

# TRAINING OPTIMIZATION FUNCTION
# ----------------------------------


def optimize_model(phase):

    
    
    # print("len memory: ", len(memory))
    # print("batch: ", BATCH_SIZE)
    if phase == 'train':
        if len(memory_train) < REPLAY_MEMORY:
        	return
        
        t_batch_size = min(len(memory_train),BATCH_SIZE)
        transitions, index = memory_train.sample(t_batch_size)  
    else:
        if len(memory_val) < REPLAY_MEMORY:
        	return
        
        t_batch_size = min(len(memory_val),BATCH_SIZE)
        transitions, _ = memory_val.sample(t_batch_size) 
    batch = Transition(*zip(*transitions))

    state_batch = torch.stack(batch.state,0)
    action_batch = torch.tensor(batch.action, device= device).unsqueeze(-1)
    reward_batch = torch.tensor(batch.reward, device=device)
    hidden_batch = torch.cat(batch.hidden)
    hidden_batch = (hidden_batch[:,0,:][None,:,:].contiguous(),hidden_batch[:,1,:][None,:,:].contiguous())

    if IMPORTANCE_SAMPLING:
        importance_sampling_batch = torch.tensor(batch.importance_sampling, device = device)

    hidden_behavioral = hidden_batch
    hiden_target = hidden_batch 

    if phase == 'val':
        with torch.no_grad():
            q_behavioral, hidden_behavioral = policy_net(state_batch,hidden_behavioral)
    else:
        q_behavioral, hidden_behavioral = policy_net(state_batch,hidden_behavioral)
    with torch.no_grad():
        q_target, hiden_target = target_net(state_batch,hiden_target)

    # st_act_behaviroal = torch.gather(q_behavioral, -1, action_batch)
    # st_act_target = torch.gather(q_target, -1, action_batch)
    
    # st_act_behaviroal = st_act_behaviroal[:,1:N_STEP,:]
    # st_act_target = st_act_target[:,1:N_STEP,:]

    G = torch.zeros(reward_batch.shape[0], device=device)
    if IMPORTANCE_SAMPLING:
        rho = torch.ones(importance_sampling_batch.shape[0],device=device)
    for idx_batch in range(reward_batch.shape[0]):
        if IMPORTANCE_SAMPLING:
            for idx_n_step, val_samp in enumerate(importance_sampling_batch[idx_batch,1:N_STEP]):
                rho[idx_batch] *= val_samp
        
        for idx_G,reward in enumerate(reward_batch[idx_batch,0:N_STEP]):
            G[idx_batch]+= (GAMMA**idx_G)*reward 
            # pdb.set_trace()

    # q_learning 
    q_values_expected = q_target[-1].max().detach()
    q_values_expected,_ = torch.max(q_target[:,-1,:],1)

    
    expected_state_action_values = (q_values_expected * (GAMMA**N_STEP)) + G #Get Q value for current state as R + Q(s')

    state_action_values = torch.gather(q_behavioral[:,0,:], -1, action_batch[:,0,:])
    
    criterion = nn.SmoothL1Loss(reduction='none') #MSE

    # loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1)) ####################
    # priority weights Replay Memory
    loss_each = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    loss = torch.mean(loss_each)
    # pdb.set_trace()


    # pdb.set_trace()
    
    if IMPORTANCE_SAMPLING:
        loss = criterion(rho*state_action_values, rho*expected_state_action_values.unsqueeze(1))
    episode_loss.append(loss.detach().item())

    # print('phase: ', phase)
    if phase == 'train':
        # print('phase: ',phase)
        # pdb.set_trace()
        # memory_train.plot_priorities(save_path_memory)
        # memory_train.plot_batch_priorities(index, save_path_memory)
        # pdb.set_trace()
        memory_train.update_priorities(index,abs(loss_each).squeeze(1).detach().cpu().numpy())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        # for param in policy_net.parameters():
        #     param.grad.data.clamp_(-1,1)
        # for p in policy_net.parameters():
        #     p.register_hook(lambda grad: torch.clamp(grad, -1, 1))

        optimizer.step()
        

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# memory_train = ReplayMemory(REPLAY_MEMORY)
# memory_val = ReplayMemory(REPLAY_MEMORY)

memory_train = PrioritizedReplayMemory(REPLAY_MEMORY)
memory_val = PrioritizedReplayMemory(REPLAY_MEMORY)

print_setup(args)

# ----------------------------------

# TRAINING LOOP
# ----------------------------------

print("\nTraining...")
print("_"*30)



steps_done = 0 

t1 = time.time() #Tik
decision_cont = 0




epoch_loss = []
total_loss_epoch_train =[]
total_time_execution_epoch_train = []
total_reward_epoch_train = []
total_reward_energy_epoch_train = []
total_reward_time_epoch_train = []

total_CA_intime_epoch_train = []
total_CA_late_epoch_train = []
total_IA_intime_epoch_train = []
total_IA_late_epoch_train = []
total_UAC_intime_epoch_train = []
total_UAC_late_epoch_train = []
total_UAI_intime_epoch_train = []
total_UAI_late_epoch_train = []
total_CI_epoch_train = []
total_II_epoch_train = []

total_loss_epoch_val =[]
total_time_execution_epoch_val = []
total_reward_epoch_val = []
total_reward_energy_epoch_val = []
total_reward_time_epoch_val = []

total_CA_intime_epoch_val = []
total_CA_late_epoch_val = []
total_IA_intime_epoch_val = []
total_IA_late_epoch_val = []
total_UAC_intime_epoch_val = []
total_UAC_late_epoch_val = []
total_UAI_intime_epoch_val = []
total_UAI_late_epoch_val = []
total_CI_epoch_val = []
total_II_epoch_val = []

#12345
total_UA_related_epoch_train = []
total_UA_unrelated_epoch_train = []
total_UA_related_epoch_val = []
total_UA_unrelated_epoch_val = []

total_G_total_epoch_train= []
total_G_energy_epoch_train = []
total_G_time_epoch_train = []
total_G_total_epoch_val= []
total_G_energy_epoch_val = []
total_G_time_epoch_val = []

prev_decision_rate = 1


# Get minimum and maximum time from dataset
video_min_times = []
video_human_times = []

# videos = glob.glob(train_videos)  
#GET VIDEO TIME AND OPTIMAL TIME (MIN)
for video in train_videos:
    path_video = video + '/human_times'
    human_times = np.load(path_video, allow_pickle=True)  
    min_time = human_times['min']
    human_time = human_times['human_time']
    
    video_human_times.append(human_time)
    video_min_times.append(min_time)

minimum_time = sum(video_min_times)
healthy_human_time = sum(video_human_times)    

print(minimum_time)
print(healthy_human_time)

cont_actions_val = ROBOT_CONT_ACTIONS_MEANINGS.copy()
cont_actions_train = ROBOT_CONT_ACTIONS_MEANINGS.copy()

total_idle_epoch_train = []
total_idle_epoch_val = []

list_actions = []
list_rewards = []
num_optim = 0
i_epoch = -1 
# MODE= ['train', 'val']
while i_epoch < NUM_EPOCH:
# for i_epoch in range (LOAD_EPISODE,NUM_EPOCH):
    i_epoch += 1
    flag_break = False
    # if len(memory_train) == REPLAY_MEMORY and len(memory_val) == REPLAY_MEMORY:
    if len(memory_train) == REPLAY_MEMORY:
        # MODE= ['train', 'val']
        steps_done += 1  
        scheduler.step()
    else:
        # if len(memory_train) == REPLAY_MEMORY:
        #     MODE = ['val']
        i_epoch -= 1
        flag_break = True
    decision_index_histogram_TRAIN = []
    decision_action_index_histogram_TRAIN = []

    good_reward_TRAIN = []
    good_reward_action_TRAIN = []
    good_reward_noaction_TRAIN = []
    bad_reward_TRAIN = []

    decision_index_histogram_VAL = []
    decision_action_index_histogram_VAL = []

    good_reward_VAL = []
    good_reward_action_VAL = []
    good_reward_noaction_VAL = []
    bad_reward_VAL = []
    # Each epoch has a training and validation phase
    print("| ----------- EPOCH " + str(i_epoch) + " ----------- ")
    
    for phase in ['train', 'val']:
    # for phase in MODE:
        total_loss = []
        total_reward = []
        total_reward_energy_ep = []
        total_reward_time_ep = []
        total_reward_error_pred = []
       
        total_G_total_ep = []
        total_G_energy_ep = []
        total_G_time_ep = []
        total_CA_intime = []
        total_CA_late = []
        total_IA_intime = []
        total_IA_late = []
        total_UAC_intime = []
        total_UAC_late = []
        total_UAI_intime = []
        total_UAI_late = []
        total_CI = []
        total_II = []
        
        #123456
        total_UA_related = []
        total_UA_unrelated = []
        
        total_idle = []
        #total_minimum_time_execution_epoch = []
        #total_maximum_time_execution_epoch = []
        total_interaction_time_epoch = []
        
        videos_mejorables = []
        total_times_execution = []
        
            
        if phase == 'train':
            policy_net.train()  # Set model to training mode
            target_net.eval()
        else:
            policy_net.eval()   # Set model to evaluate mode
            target_net.eval()
        for i_episode in range(0, NUM_EPISODES):
            if(args.display): print("| EPISODE #", i_episode , end='\n')
            else: print("| EPISODE #", i_episode , end='\r')
           
            # print("| EPISODE #", i_episode)
            state = torch.tensor(env.reset(), dtype=torch.float, device=device).unsqueeze(0)
            
            episode_loss = []
            done = False
            to_optim = True
            
            decision_state = state
            
            
            reward_energy_ep = 0
            reward_time_ep = 0
            error_pred_ep = 0
            total_pred_ep = 0
            G_total_ep = 0
            G_energy_ep = 0
            G_time_ep = 0
            

            n_decision_state = []
            n_action = []
            n_reward = []
            # n_hidden = []
            n_imp_sampling = []
            temporal_buffer = circular_buffer()
            G_total = circular_buffer()
            G_energy = circular_buffer()
            G_time = circular_buffer()
            # initialize the hidden state. 
            # every time a new recipe is started the hidden latent state must be initialized
            
            if device.type == 'cuda':
                hidden = (torch.randn(1,hidden_size_LSTM).cuda(),
                          torch.randn(1,hidden_size_LSTM).cuda())
            else:
                hidden = (torch.randn(hidden_size_LSTM),
                          torch.randn(hidden_size_LSTM))
            hidden_train = hidden
            hidden_val = hidden 
            # print('estamos en phase: ',phase)
            for t in count(): 

                decision_cont += 1
                n_hidden = hidden
                if phase == 'train':
                    hidden_train = n_hidden 
                else:
                    hidden_val = n_hidden
                action, flag_decision, prev_decision_rate, hidden = action_rate(decision_cont, torch.tensor(env.state, dtype=torch.float, device=device).unsqueeze(0), hidden_train, hidden_val, phase, prev_decision_rate)
               
                if flag_decision: 
                    action_ = action
                    action = action.item()
                    frame_decision = env.get_frame()
                    action_idx = env.get_action_idx()
                    annotations = env.get_annotations()
                    decision_cont = 0
                    if IMPORTANCE_SAMPLING:
                        imp_sampling = importance_sampling (state, hidden, action) # no se puede enviar ese hidden
                    if to_optim:
                        decision_state = torch.tensor(env.state, dtype=torch.float, device=device).unsqueeze(0)
                        to_optim = False
                        
                
            
                array_action = [action,flag_decision]
                next_state_, reward, done, optim, flag_pdb, reward_time, reward_energy, hri_time, correct_action, type_threshold, error_pred, total_pred = env.step([array_action,array_conf])

            
                reward = torch.tensor([reward], device=device)
                next_state = torch.tensor(env.state, dtype=torch.float, device=device).unsqueeze(0)
                reward_energy_ep += reward_energy
                reward_time_ep += reward_time
                error_pred_ep += error_pred
                total_pred_ep += total_pred
                
                # print(' Flag decision after: ', flag_decision)
                if flag_decision:
                    reward = torch.tensor([reward], device=device)
                    if IMPORTANCE_SAMPLING:
                        data = [n_hidden,decision_state, action_,reward, imp_sampling]
                    else:
                        data = [n_hidden,decision_state, action_,reward]
                    temporal_buffer.append(data)
                    # print(action_)
                prob_save = False
                if random.random() < PROB_SAVE: # un 10% del tiepo se guardan en memoria datos de no optimizar, casos de no accion
                    prob_save = True
                # if i_epoch == 0:
                #     minimum_time += total_minimum_time_execution
                if optim or prob_save: #Only train if we have taken an action (f==30)                  
                    
                    # tengo que guaradr el temporal buffer pero  solo me quedo con el primer hidden 
                    if len(temporal_buffer) == N_STEP:
                        data_to_save = list(temporal_buffer)
                        hidden_0 = temporal_buffer[0][0]
                        n_decision_state = [item[1] for item in data_to_save]
                        n_action = [item[2] for item in data_to_save]
                        n_reward = [item[3] for item in data_to_save]
                        if IMPORTANCE_SAMPLING:
                            n_imp_sampling = [item[4] for item in data_to_save]
                            if phase == 'train':
                                memory_train.push(torch.cat(n_decision_state),n_action,n_reward,torch.stack(hidden_0,1),n_imp_sampling)
                            else:
                                memory_val.push(torch.cat(n_decision_state),n_action,n_reward,torch.stack(hidden_0,1),n_imp_sampling)
                        else:
                            if phase == 'train':
                                memory_train.push(torch.cat(n_decision_state),n_action,n_reward,torch.stack(hidden_0,1))
                            else:
                                memory_val.push(torch.cat(n_decision_state),n_action,n_reward,torch.stack(hidden_0,1))
                        
                        if optim and i_epoch >-1:
                            to_optim = True

                            G_total.append(reward.detach().cpu().numpy())
                            G_energy.append(reward_energy)
                            G_time.append(reward_time)
                            if len(G_total) == N_STEP:
                                for idx_G in range(0,N_STEP):
                                    G_total_ep+= (GAMMA**idx_G)*G_total[idx_G] 
                                    G_energy_ep+= (GAMMA**idx_G)*G_energy[idx_G] 
                                    G_time_ep+= (GAMMA**idx_G)*G_time[idx_G] 
                            optimize_model(phase)
                            num_optim += 1
                    
                if not done: 
                    state = next_state
        
                else: 
                    next_state = None 
                
                if done: 
                    
                    if episode_loss: 
                    	total_loss.append(mean(episode_loss))
                    	
                    	
                    total_reward.append(env.get_total_reward())
                    total_reward_energy_ep.append(reward_energy_ep)
                    total_reward_time_ep.append(reward_time_ep)
                    total_reward_error_pred.append(error_pred_ep/total_pred_ep)                    
                    total_CA_intime.append(env.CA_intime)
                    total_CA_late.append(env.CA_late)
                    total_IA_intime.append(env.IA_intime)
                    total_IA_late.append(env.IA_late)
                    total_UAC_intime.append(env.UAC_intime)
                    total_UAC_late.append(env.UAC_late)
                    total_UAI_intime.append(env.UAI_intime)
                    total_UAI_late.append(env.UAI_late)
                    total_CI.append(env.CI)
                    total_II.append(env.II)
                    total_G_total_ep.append(G_total_ep)
                    total_G_energy_ep.append(G_energy_ep)
                    total_G_time_ep.append(G_time_ep)

                    #Interaction
                    total_interaction_time_epoch.append(hri_time)
                    
                    #123456
                    total_UA_related.append(env.UA_related)
                    total_UA_unrelated.append(env.UA_unrelated)
                    total_idle.append(env.anticipation) 
                    
                    #Baseline times
                    #total_minimum_time_execution_epoch.append(min_time) #Minimum possible time
                    #total_maximum_time_execution_epoch.append(max_time) #Human max time -> no HRI

                    break #Finish episode
           
            if phase == 'train':
                # target_net_state_dict = target_net.state_dict()
                # policy_net_state_dict = policy_net.state_dict()
                # for key in policy_net_state_dict:
                #     target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                # target_net.load_state_dict(target_net_state_dict)
                # if i_episode % TARGET_UPDATE == 0: #Copy the Policy Network parameters into Target Network
                if num_optim % TARGET_UPDATE == 0:
    
                    target_net.load_state_dict(policy_net.state_dict())
                
        if flag_break == True:
            break
                            
        #total_time_video = list(list(zip(*total_times_execution))[0])
        #total_time_interaction = list(list(zip(*total_times_execution))[1])
        #minimum_time = sum(total_minimum_time_execution_epoch)
        #maximum_time = sum(total_maximum_time_execution_epoch)
        interaction_time = sum(total_interaction_time_epoch)
        
        #print("\n\n\n\nIN TRAIN, minimum: ", minimum_time)
     
        
        # data = {'video': healthy_human_time,
        # 'interaction': interaction_time,
        # 'CA_intime': total_CA_intime,
        # 'CA_late':total_CA_late,
        # 'IA_intime': total_IA_intime,
        # 'IA_late':total_IA_late,

        # 'UAC_intime': total_UAC_intime,
        # 'UAC_late': total_UAC_late,
        # 'UAI_intime': total_UAI_intime,
        # 'UAI_late': total_UAI_late,
        # 'CI': total_CI,
        # 'II': total_II,
        
        # 'prediction error': total_reward_error_pred
        # }
        
        
        # if i_epoch == 0: 
        #     df = pd.DataFrame(data)
        # else:
        #     df_new = pd.DataFrame(data)
        #     df = pd.concat([df,df_new])
        
            
        if phase == 'train':
            
            
            transitions, _ = memory_train.sample(REPLAY_MEMORY) 
            batch = Transition(*zip(*transitions))
            action_batch = torch.tensor(batch.action, device= device).squeeze().detach().cpu().numpy()
            reward_batch = torch.tensor(batch.reward, device= device).squeeze().detach().cpu().numpy()
            list_actions.append(action_batch)
            list_rewards.append(reward_batch)
            
            if i_epoch % round(NUM_EPOCH*0.01) == 0:
                # print(PRETRAINED)
                model_name = 'model_' + str(i_epoch) + '.pt'
                print("Saving model at ", os.path.join(path, model_name))
                torch.save({
                'model_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': i_epoch,
                'loss': total_loss,
                'steps': steps_done            
                }, os.path.join(path, model_name))
                
                if TEMPORAL_CTX:
                    ROBOT_EXECUTION_TIMES  = env.get_robot_execution_times()
                    with open(path+'/robot_execution_times', 'wb') as handle:
                        pickle.dump(ROBOT_EXECUTION_TIMES, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    
                # create data
            if i_epoch % 20 == 0:
                
                
                G = np.zeros(reward_batch.shape[0])
                for idx_batch in range(reward_batch.shape[0]):
                    for idx_G,reward in enumerate(reward_batch[idx_batch,0:N_STEP]):
                        G[idx_batch]+= (GAMMA**idx_G)*reward 
                    
                actions = np.reshape(action_batch,action_batch.shape[0]*action_batch.shape[1])
                y1 = sorted(Counter(actions).items())
                y1 = [el[1] for el in y1]
                
                y1_total = sum(y1)
                y1_norm = [item/y1_total for item in y1]
                x = ('0','1','2','3','4','5')                
                width = 0.6  # the width of the bars: can also be len(x) sequence
                
                fig, ax = plt.subplots()
                p = ax.bar(x,y1, width, label='Epoch '+str(i_epoch))
                ax.bar_label(p, label_type='center')
                ax.set_title("Actions saved in memory in epoch "+str(i_epoch))
                ax.set_ylabel("Action counts")
                ax.set_xlabel("Action")
                ax.legend()
                fig.savefig(save_path_hist+'/Histogram_actions_'+str(i_epoch)+'.jpg')
                
                fig, ax = plt.subplots()
                p = ax.bar(x,y1_norm, width, label='Epoch '+str(i_epoch))
                ax.bar_label(p, label_type='center')
                ax.set_title("Actions saved in memory (sum to one) in epoch "+str(i_epoch))
                ax.set_ylabel("Instances")
                ax.set_xlabel("Action")
                ax.legend()
                fig.savefig(save_path_hist+'/Histogram_actions_norm'+str(i_epoch)+'.jpg')
                
                first_no_action_idx = np.where(action_batch[:,0]==5)[0]
                first_action_idx = np.where(action_batch[:,0]!=5)[0]
                G_no_action = G[first_no_action_idx]
                G_action = G[first_action_idx]
                
                x = ('NAC','NAI','AC','AI') 
                NAC = sum(map(lambda i: i == 0, G_no_action))
                NAI = sum(map(lambda i: i != 0, G_no_action))
                AC = sum(map(lambda i: i == 0, G_action))
                AI =sum(map(lambda i: i != 0, G_action))
                
                total_G = NAC + NAI + AC + AI 
                
                NAC_NORM = NAC / total_G
                NAI_NORM = NAI / total_G
                AC_NORM = AC / total_G
                AI_NORM = AI / total_G
                
                fig, ax = plt.subplots()
                p = ax.bar(x,[NAC,NAI,AC,AI], width, label='Epoch '+str(i_epoch))
                ax.bar_label(p, label_type='center')
                ax.set_title("Grouping according to G obtained based on \nfirst action taken of the N_STEP in epoch "+str(i_epoch))
                ax.set_ylabel("Instances")
                ax.set_xlabel(" First action of the N_STEP & G obtained")
                ax.legend()
                fig.savefig(save_path_hist+'/Histogram_G_actions_'+str(i_epoch)+'.jpg')
                
                fig, ax = plt.subplots()
                p = ax.bar(x,[NAC_NORM,NAI_NORM,AC_NORM,AI_NORM], width, label='Epoch '+str(i_epoch))
                ax.bar_label(p, label_type='center')
                ax.set_title("Grouping according to G obtained based on \nfirst action taken of the N_STEP in epoch "+str(i_epoch))
                ax.set_ylabel("Instances")
                ax.set_xlabel(" First action of the N_STEP & G obtained")
                ax.legend()
                fig.savefig(save_path_hist+'/Histogram_G_actions_norm'+str(i_epoch)+'.jpg')
                
                no_action_idx = np.where(action_batch==5)
                coordinates_no_action = np.vstack((no_action_idx[0], no_action_idx[1])).T
                action_idx = np.where(action_batch!=5)
                coordinates_action = np.vstack((action_idx[0], action_idx[1])).T
                reward_no_action = []
                reward_action = []
                # pdb.set_trace()
                # Iterar a través de las coordenadas y agregar los valores correspondientes a la lista de valores deseados
                for coor in coordinates_no_action:
                    row, col = coor
                    reward = reward_batch[row][col]
                    reward_no_action.append(reward)
                    
                for coor in coordinates_action:
                    row, col = coor
                    reward = reward_batch[row][col]
                    reward_action.append(reward)
                    
                    
                NAC = sum(map(lambda i: i == 0, reward_no_action))
                NAI = sum(map(lambda i: i != 0, reward_no_action))
                AC = sum(map(lambda i: i == 0, reward_action))
                AI =sum(map(lambda i: i != 0, reward_action))
                
                total_r = NAC + NAI + AC + AI 
                
                NAC_NORM = NAC / total_r
                NAI_NORM = NAI / total_r
                AC_NORM = AC / total_r
                AI_NORM = AI / total_r
                
                fig, ax = plt.subplots()
                p = ax.bar(x,[NAC,NAI,AC,AI], width, label='Epoch '+str(i_epoch))
                ax.bar_label(p, label_type='center')
                ax.set_title("Grouping according to R obtained based on \nfirst action taken of the N_STEP in epoch "+str(i_epoch))
                ax.set_ylabel("Instances")
                ax.set_xlabel(" First action of the N_STEP & R obtained")
                ax.legend()
                fig.savefig(save_path_hist+'/Histogram_R_actions_'+str(i_epoch)+'.jpg')
                
                fig, ax = plt.subplots()
                p = ax.bar(x,[NAC_NORM,NAI_NORM,AC_NORM,AI_NORM], width, label='Epoch '+str(i_epoch))
                ax.bar_label(p, label_type='center')
                ax.set_title("Grouping according to R obtained based on \nfirst action taken of the N_STEP in epoch "+str(i_epoch))
                ax.set_ylabel("Instances")
                ax.set_xlabel(" First action of the N_STEP & R obtained")
                ax.legend()
                fig.savefig(save_path_hist+'/Histogram_R_actions_norm'+str(i_epoch)+'.jpg')
                
                
                # pdb.set_trace()
                
                # list_actions_filtered = list_actions[i_epoch-4:]
                # list_rewards_filtered = list_rewards[i_epoch-4:]
                
                # do_nothing_index_y1 = np.where(list_actions_filtered[0]==5)[0]
                # do_nothing_index_y2 = np.where(list_actions_filtered[1]==5)[0]
                # do_nothing_index_y3 = np.where(list_actions_filtered[2]==5)[0]
                # do_nothing_index_y4 = np.where(list_actions_filtered[3]==5)[0]

                # do_nothing_reward_y1 = list_rewards_filtered[0][do_nothing_index_y1]
                # do_nothing_reward_y2 = list_rewards_filtered[1][do_nothing_index_y2]
                # do_nothing_reward_y3 = list_rewards_filtered[2][do_nothing_index_y3]
                # do_nothing_reward_y4 = list_rewards_filtered[3][do_nothing_index_y4]
                
                # # get times do nothing get positive reward (0) or different reward (bad)
                # cont_0 = 0
                # cont_dif_0 = 0
                # for rew in do_nothing_reward_y1:
                #     if rew == 0:
                #         cont_0 += 1
                #     else:
                #         cont_dif_0 += 1
                
                # do_nothing_rew_type_y1 = [cont_0, cont_dif_0]
                # cont_0 = 0
                # cont_dif_0 = 0
                # for rew in do_nothing_reward_y2:
                #     if rew == 0:
                #         cont_0 += 1
                #     else:
                #         cont_dif_0 += 1
                
                # do_nothing_rew_type_y2 = [cont_0, cont_dif_0]
                # cont_0 = 0
                # cont_dif_0 = 0
                # for rew in do_nothing_reward_y3:
                #     if rew == 0:
                #         cont_0 += 1
                #     else:
                #         cont_dif_0 += 1
                
                # do_nothing_rew_type_y3 = [cont_0, cont_dif_0]
                # cont_0 = 0
                # cont_dif_0 = 0
                # for rew in do_nothing_reward_y4:
                #     if rew == 0:
                #         cont_0 += 1
                #     else:
                #         cont_dif_0 += 1
                
                # do_nothing_rew_type_y4 = [cont_0, cont_dif_0]
                
                # y1 = sorted(Counter(list_actions_filtered[0]).items())
                # y2 = sorted(Counter(list_actions_filtered[1]).items())
                # y3 = sorted(Counter(list_actions_filtered[2]).items())
                # y4 = sorted(Counter(list_actions_filtered[3]).items())
                 
                # y1 = [el[1] for el in y1]
                # y2 = [el[1] for el in y2]
                # y3 = [el[1] for el in y3]
                # y4 = [el[1] for el in y4]

                # # pdb.set_trace()
                # # plot bars in stack manner
                
                # x = ('0','1','2','3','4','5')
                # action_counts = {
                #     'Epoch '+str(i_epoch-3): y1,
                #     'Epoch '+str(i_epoch-2): y2,
                #     'Epoch '+str(i_epoch-1): y3,
                #     'Epoch '+str(i_epoch): y4
                #     }
                
                # width = 0.6  # the width of the bars: can also be len(x) sequence
                
                # fig, ax = plt.subplots()
                # bottom = np.zeros(len(x))
                # for act, act_count in action_counts.items():
                #     p = ax.bar(x, act_count, width, label=act, bottom=bottom)
                #     bottom += act_count
                
                #     ax.bar_label(p, label_type='center')
                
                # ax.set_title("Actions taken in 4 epochs")
                # ax.set_ylabel("Action counts")
                # ax.set_xlabel("Action")
                # ax.legend()
                # fig.savefig(save_path_hist+'/Histogram_actions_'+str(i_epoch)+'.jpg')
                
                
                
                # x1 = ('0','!=0')
                # action_counts = {
                #     'Epoch '+str(i_epoch-3): do_nothing_rew_type_y1,
                #     'Epoch '+str(i_epoch-2): do_nothing_rew_type_y2,
                #     'Epoch '+str(i_epoch-1): do_nothing_rew_type_y3,
                #     'Epoch '+str(i_epoch): do_nothing_rew_type_y4
                #     }
                # width = 0.6  # the width of the bars: can also be len(x) sequence
                
                # fig1, ax1 = plt.subplots()
                # bottom = np.zeros(len(x1))
                # for act, act_count in action_counts.items():
                #     p = ax1.bar(x1, act_count, width, label=act, bottom=bottom)
                #     bottom += act_count
                
                #     ax1.bar_label(p, label_type='center')
                
                # ax1.set_title("Type reward of do nothing action obtained")
                # ax1.set_ylabel("Action counts")
                # ax1.set_xlabel("Reward")
                # ax1.legend()
                
                # # plt.show()
                # fig1.savefig(save_path_hist+'/Histogram_reward_do_nothing_'+str(i_epoch)+'.jpg')
                
                # pdb.set_trace()


            ex_rate.append(EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY))

            total_loss_epoch_train.append(sum(total_loss))
            total_reward_epoch_train.append(sum(total_reward))
            total_reward_energy_epoch_train.append(sum(total_reward_energy_ep))
            total_reward_time_epoch_train.append(sum(total_reward_time_ep))
            
            total_G_total_epoch_train.append(sum(total_G_total_ep))
            total_G_energy_epoch_train.append(sum(total_G_energy_ep))
            total_G_time_epoch_train.append(sum(total_G_time_ep))

            #total_time_execution_epoch_train.append(sum(total_time_interaction))
            total_time_execution_epoch_train.append(interaction_time)

            total_CA_intime_epoch_train.append(sum(total_CA_intime))
            total_CA_late_epoch_train.append(sum(total_CA_late))
            total_IA_intime_epoch_train.append(sum(total_IA_intime))
            total_IA_late_epoch_train.append(sum(total_IA_late))
            total_UAC_intime_epoch_train.append(sum(total_UAC_intime))
            total_UAC_late_epoch_train.append(sum(total_UAC_late))
            total_UAI_intime_epoch_train.append(sum(total_UAI_intime))
            total_UAI_late_epoch_train.append(sum(total_UAI_late))
            total_CI_epoch_train.append(sum(total_CI))
            total_II_epoch_train.append(sum(total_II))
            
            #123456
            total_UA_related_epoch_train.append(sum(total_UA_related))
            total_UA_unrelated_epoch_train.append(sum(total_UA_unrelated))
            total_idle_epoch_train.append(np.mean(total_idle))
            
            #123456 -> Add (un)related
            total_results_train = [total_CA_intime_epoch_train,total_CA_late_epoch_train,total_IA_intime_epoch_train,
            total_IA_late_epoch_train,
            total_UAC_intime_epoch_train,
            total_UAC_late_epoch_train,
            total_UAI_intime_epoch_train,
            total_UAI_late_epoch_train,
            total_CI_epoch_train,
            total_II_epoch_train,
            total_UA_related_epoch_train, #123456
            total_UA_unrelated_epoch_train] #123456
            

 
            #PLOT TRAIN
            if i_epoch % round(NUM_EPOCH*0.01) == 0:
                plot_each_epoch(i_epoch, phase,save_path,
                minimum_time,
                total_results_train,
                total_loss_epoch_train,
                total_reward_epoch_train,
                healthy_human_time,
                total_time_execution_epoch_train,
                total_reward_energy_epoch_train,
                total_reward_time_epoch_train,
                total_G_total_epoch_train,
                total_G_energy_epoch_train,
                total_G_time_epoch_train,
                ex_rate)
            
            
            
            
            # if i_epoch == NUM_EPOCH-1:
            data_train = {
            'CA_intime': total_CA_intime_epoch_train,
            'CA_late':total_CA_late_epoch_train,
            'IA_intime': total_IA_intime_epoch_train,
            'IA_late':total_IA_late_epoch_train,
            'UAC_intime': total_UAC_intime_epoch_train,
            'UAC_late': total_UAC_late_epoch_train,
            'UAI_intime': total_UAI_intime_epoch_train,
            'UAI_late': total_UAI_late_epoch_train,
            'CI': total_CI_epoch_train,
            'II': total_II_epoch_train,
            'prediction error': np.mean(total_reward_error_pred)
            }

            #df_train = pd.DataFrame(data_train)
            #df_train.to_csv(save_path+'/data_train.csv')



            #print("\n(train) PREDICTION ERROR: %.2f%%" %(np.mean(total_reward_error_pred)*100))
        elif phase=='val':
            # print(len(total_loss))
            # pdb.set_trace()
            total_loss_epoch_val.append(sum(total_loss))
            total_reward_epoch_val.append(sum(total_reward))
            total_reward_energy_epoch_val.append(sum(total_reward_energy_ep))
            total_reward_time_epoch_val.append(sum(total_reward_time_ep))

            total_G_total_epoch_val.append(sum(total_G_total_ep))
            total_G_energy_epoch_val.append(sum(total_G_energy_ep))
            total_G_time_epoch_val.append(sum(total_G_time_ep))
            #total_time_execution_epoch_val.append(sum(total_time_interaction))
            total_time_execution_epoch_val.append(interaction_time)

            total_CA_intime_epoch_val.append(sum(total_CA_intime))
            total_CA_late_epoch_val.append(sum(total_CA_late))
            total_IA_intime_epoch_val.append(sum(total_IA_intime))
            total_IA_late_epoch_val.append(sum(total_IA_late))
            total_UAC_intime_epoch_val.append(sum(total_UAC_intime))
            # print("total_UAC_intime_epoch_val: ", total_UAC_intime_epoch_val)
            
            total_UAC_late_epoch_val.append(sum(total_UAC_late))
            total_UAI_intime_epoch_val.append(sum(total_UAI_intime))
            # print("total_UAI_intime_epoch_val: ", total_UAI_intime_epoch_val)
            total_UAI_late_epoch_val.append(sum(total_UAI_late))
            total_CI_epoch_val.append(sum(total_CI))
            total_II_epoch_val.append(sum(total_II))
            
            total_idle_epoch_val.append(np.mean(total_idle))
            #123456
            total_UA_related_epoch_val.append(sum(total_UA_related))
            total_UA_unrelated_epoch_val.append(sum(total_UA_unrelated))
            # -----
            
            #123456 -> Add to validation total results
            total_results = [total_CA_intime_epoch_val,total_CA_late_epoch_val,total_IA_intime_epoch_val,
            total_IA_late_epoch_val,
            total_UAC_intime_epoch_val,
            total_UAC_late_epoch_val,
            total_UAI_intime_epoch_val,
            total_UAI_late_epoch_val,
            total_CI_epoch_val,
            total_II_epoch_val,
            total_UA_related_epoch_val, #123456
            total_UA_unrelated_epoch_val]
            #PLOT VALIDATION
            
            
            if i_epoch % round(NUM_EPOCH*0.01) == 0:
                plot_each_epoch(i_epoch, phase,save_path,
                minimum_time, 
                total_results,
                total_loss_epoch_val,
                total_reward_epoch_val,
                healthy_human_time,
                total_time_execution_epoch_val,
                total_reward_energy_epoch_val,
                total_reward_time_epoch_val,
                total_G_total_epoch_val,
                total_G_energy_epoch_val,
                total_G_time_epoch_val)
            
            
            
            #---------------------------------------------------------------------------------------
            
            #PLOT TOGETHER
            if i_epoch % round(NUM_EPOCH*0.01) == 0: 
                plot_each_epoch_together(i_epoch,save_path,
                minimum_time,
                total_results_train,
                total_loss_epoch_train,
                total_reward_epoch_train,
                healthy_human_time,
                total_time_execution_epoch_train,
                total_reward_energy_epoch_train,
                total_reward_time_epoch_train,
                ex_rate,
                total_results,
                total_loss_epoch_val,
                total_reward_epoch_val,
                total_time_execution_epoch_val,
                total_reward_energy_epoch_val,
                total_reward_time_epoch_val,
                total_G_total_epoch_train,
                total_G_energy_epoch_train,
                total_G_time_epoch_train,
                total_G_total_epoch_val,
                total_G_energy_epoch_val,
                total_G_time_epoch_val)
            
            # if i_epoch == NUM_EPOCH-1:
            data_val = {
            'CA_intime': total_CA_intime_epoch_val,
            'CA_late':total_CA_late_epoch_val,
            'IA_intime': total_IA_intime_epoch_val,
            'IA_late':total_IA_late_epoch_val,
            'UAC_intime': total_UAC_intime_epoch_val,
            'UAC_late': total_UAC_late_epoch_val,
            'UAI_intime': total_UAI_intime_epoch_val,
            'UAI_late': total_UAI_late_epoch_val,
            'CI': total_CI_epoch_val,
            'II': total_II_epoch_val,
            'prediction error': np.mean(total_reward_error_pred)
            }
            
    
            
            

t2 = time.time() - t1 #Tak


print("\nTraining completed in {:.1f}".format(t2), "seconds.\n")
if PRETRAINED:
    with open(path +'/model_used.txt', 'w') as f: f.write(path_model.split('/')[-1])

# CONFIG CONFIGURATION FILE 
file1 = open(path+"/CONFIGURATION.txt","a")

CONFIG_PARAM = [N_STEP,REPLAY_MEMORY,NUM_EPOCH,NUM_EPISODES,BATCH_SIZE,GAMMA,EPS_START,EPS_END,EPS_DECAY,TARGET_UPDATE,LR,POSITIVE_REWARD,NO_ACTION_PROBABILITY,FACTOR_ENERGY_PENALTY,DECISION_RATE, SPEED_ROBOT, TEMPORAL_CTX]
CONFIG_PARAM_NAME = ["N_STEP","REPLAY_MEMORY","NUM_EPOCH","NUM_EPISODES","BATCH_SIZE","GAMMA","EPS_START","EPS_END","EPS_DECAY","TARGET_UPDATE","LR","POSITIVE_REWARD","NO_ACTION_PROBABILITY","FACTOR_ENERGY_PENALTY","DECISION_RATE", "SPEED","TEMPORAL_CTX"]
for idx,conf in enumerate(CONFIG_PARAM):
    file1.write(CONFIG_PARAM_NAME[idx]+": ")
    file1.write(str(conf)+', ')
file1.close()

if TEMPORAL_CTX:
    ROBOT_EXECUTION_TIMES  = env.get_robot_execution_times()
    with open(path+'/robot_execution_times', 'wb') as handle:
        pickle.dump(ROBOT_EXECUTION_TIMES, handle, protocol=pickle.HIGHEST_PROTOCOL)
    