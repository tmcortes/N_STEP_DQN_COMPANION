
"""

This script runs the test for a model that was saved in different training epochs.
The only required parameter is the "experiment_name", which should be the name of the folder where the different checkpoints were saved.
The checkpoints are named as "model_#epoch.pt". 

"""




import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy 
from itertools import count
from statistics import mean
import os
import glob
import collections
import torch
import torch.nn as nn
import pdb
from DQN import DQN, ReplayMemory, Transition, init_weights 
from config import print_setup
import config as cfg
from aux import *
from natsort import natsorted, ns
import re
from generate_history import *

#ARGUMENTS
import config as cfg
import argparse

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str, default=cfg.EXPERIMENT_NAME, help="(str) Name of the experiment. Used to name the folder where the model was saved during training. For example: my_first_DQN.")

parser.add_argument('--root', type=str, default=cfg.ROOT, help="(str) Name of the root folder for the saving of checkpoints. Parent folder of EXPERIMENT_NAME folders. For example: ./Checkpoints/")
#parser.add_argument('--num_episodes', type=int, default=1000, help="(int) Number of episodes.")
parser.add_argument('--eps_test', type=float, default=0.0, help="(float) Exploration rate for the action-selection during test. For example: 0.05")
parser.add_argument('--display', action='store_true', default=False, help="Display environment info as [Current state, action taken, transitioned state, immediate reward, total reward].")
parser.add_argument('--cuda', action='store_true', default=True, help="Use GPU if available.")
parser.add_argument('--train', action='store_true', default=False, help="Run test over the training set.")
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
parser.add_argument('--reactive', action='store_true', default=cfg.REACTIVE, help="Run test of a reactive robot")
parser.add_argument('--healthy', action='store_true', default=cfg.REACTIVE, help="Run test of a reactive robot")
parser.add_argument('--loop', action='store_true', default=cfg.LOOP, help="Run test of a reactive robot")
parser.add_argument('--num_loop', type=int, default=cfg.NUM_LOOP, help="Run test of a reactive robot")


args = parser.parse_args()


state_dic = cfg.ATOMIC_ACTIONS_MEANINGS   
action_dic = cfg.ROBOT_ACTIONS_MEANINGS 

BATCH_SIZE = args.batch_size
GAMMA = args.gamma
EPS_START = args.eps_start
EPS_END = args.eps_end
EPS_DECAY = args.eps_decay

LR = args.lr
# ENERGY_PENALTY = args.energy_penalty
SPEED_ROBOT = args.speed
# print(SPEED_ROBOT)
FACTOR_ENERGY_PENALTY = args.energy_penalty
REACTIVE = args.reactive

ROBOT_ACTION_DURATIONS = copy.deepcopy(cfg.ROBOT_AVERAGE_DURATIONS)
ROBOT_ACTION_DURATIONS.update((x, y*SPEED_ROBOT) for x, y in ROBOT_ACTION_DURATIONS.items())
# print(ROBOT_ACTION_DURATIONS)
ERROR_PROB = args.failure_rate
HEALTHY = args.healthy
array_conf = [FACTOR_ENERGY_PENALTY,ROBOT_ACTION_DURATIONS, ERROR_PROB,HEALTHY]
ONLY_RECOGNITION = cfg.ONLY_RECOGNITION
NORMALIZE_DATA = cfg.NORMALIZE_DATA
PERSONALIZATION = cfg.PERSONALIZATION
TYPE_PERSONALIZATION = cfg.TYPE_PERSONALIZATION


LOOP = args.loop
NUM_LOOP = args.num_loop
#CONFIGURATION
ROOT = args.root
EXPERIMENT_NAME = args.experiment_name


def circular_buffer():
    """
    circular buffer, will store temporary data of the actions (including idle state) 
    taken by the robot, one will be created for each video, i.e. each episode.
    """
    buffer = collections.deque(maxlen=N_STEP)
    return buffer 

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
def select_action(state, hidden):
    """
    Function that chooses which action to take in a given state based on the exploration-exploitation paradigm.
    This action will always be what is referred to as a possible action; the actions possible by the robot are 
    defined by the objects available in its environment.
    Input:
        state: (tensor) current state of the environment.
    Output:
        action: (tensor) either the greedy action (argmax Q value) from the policy network output, or a random action. 
    """
    global steps_done
    posible_actions = env.possible_actions_taken_robot()
    index_posible_actions = [i for i, x in enumerate(posible_actions) if x == 1]

    # if phase == 'val':
    with torch.no_grad():
        out, hidden = policy_net(state, hidden)
        action = post_processed_possible_actions(out,index_posible_actions)
        return action, hidden


def action_rate(decision_cont,state,hidden,prev_decision_rate):
    """
    Function that sets the rate at which the robot makes decisions.
    """
    
    # if cfg.DECISION_RATE == "random":
        
    #     if phase == 'train':
    #         if decision_cont == 1:
    #             decision_rate = random.randrange(10,150)
    #             prev_decision_rate = decision_rate
    #         else:
    #              decision_rate = prev_decision_rate
    #     else:
    #         decision_rate = 100
             
    # else:
    decision_rate = cfg.DECISION_RATE
    prev_decision_rate = " "
        
    if decision_cont % decision_rate == 0:  
        action_selected, hidden = select_action(state, hidden)
        flag_decision = True 
    else:
        action_selected = 5
        flag_decision = False
    # print("RANDOM NUMBER: ",decision_rate)
    # pdb.set_trace()
    return action_selected, flag_decision, prev_decision_rate, hidden
    







#NUM_EPISODES = args.num_episodes
EPS_TEST = args.eps_test


device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")



#TEST 
#----------------
#Environment - Custom basic environment for kitchen recipes
env = gym.make("gym_basic:basic-v0", display=args.display, test=not args.train, disable_env_checker=True)

if ONLY_RECOGNITION:
    dataset = 'dataset_pred'
else:
    dataset = 'dataset_pred_recog_tmp_ctx'
    
root_realData = "./video_annotations/"+dataset+"/*" #!
videos_realData = glob.glob(root_realData) #Folders


if cfg.PERSONALIZATION:
        with open("./video_annotations/"+cfg.PERSON+"_test.txt") as f:
            lines_x = f.readlines()
            lines = []
            for line in lines_x:
                lines.append(root_realData.split('*')[0]+line.split('\n')[0])
 
else:
    with open("./video_annotations/"+cfg.TEST_FOLD+"_test.txt") as f:
        lines_x = f.readlines()
        lines = []
        for line in lines_x:
            lines.append(root_realData.split('*')[0]+line.split('\n')[0])



        
if env.test: 

    videos_realData_test=lines
    NUM_EPISODES=len(videos_realData_test)
    videos = videos_realData_test
    # NUM_EPISODES = len(glob.glob("./video_annotations/5folds/"+cfg.TEST_FOLD+"/test/*")) #Run the test only once for every video in the testset
    print("Test set")
    # root = './video_annotations/5folds/'+cfg.TEST_FOLD+'/test/*'
else:
    train_videos = list(set(videos_realData)-set(lines))
    videos_realData = train_videos
    random.shuffle(videos_realData)
    NUM_EPISODES = len(videos_realData)
    videos = videos_realData
    
    # NUM_EPISODES = len(glob.glob("./video_annotations/5folds/"+cfg.TEST_FOLD+"/train/*"))
    print("Train set")
    # root = './video_annotations/5folds/'+cfg.TEST_FOLD+'/train/*'
    
video_max_times = []
video_min_times = []
video_human_times = []

# videos = glob.glob(root)  

#GET VIDEO TIME AND OPTIMAL TIME (MIN)
for video in videos:
    path = video + '/human_times'
    human_times = np.load(path, allow_pickle=True)  
    min_time = human_times['min']
    human_time = human_times['human_time']
    
    video_human_times.append(human_time)
    video_min_times.append(min_time)

minimum_time = sum(video_min_times)
healthy_human_time = sum(video_human_times)
    # print(annotations)
    

print("healthy_human_time: ",healthy_human_time)
# print(minimum_time)
    
n_actions = env.action_space.n
n_states = env.observation_space.n

hidden_size_LSTM = cfg.hidden_size_LSTM
policy_net = DQN(n_states, hidden_size_LSTM, n_actions).to(device)
target_net = DQN(n_states, hidden_size_LSTM, n_actions).to(device)

# policy_net = DQN(n_states, n_actions).to(device)

#LOAD MODEL from 'EXPERIMENT_NAME' folder
path = os.path.join(ROOT, EXPERIMENT_NAME)

print("PATH: ", path)

with open(path+'/CONFIGURATION.txt') as f:
    lines = str(f.readlines())
    lines = lines.split('[')[1].split(']')[0]
    lines = lines.split(',')

for idx,line in enumerate(lines):
    if 'N_STEP' in line:
        N_STEP = int(lines[idx].split('N_STEP: ')[1])
    if 'FACTOR_ENERGY_PENALTY' in line:
        FACTOR_ENERGY_PENALTY = float(lines[idx].split('FACTOR_ENERGY_PENALTY: ')[1])
    if 'DECISION_RATE' in line:
        DECISION_RATE = int(lines[idx].split('DECISION_RATE: ')[1])
        
# pdb.set_trace()
# N_STEP = int(EXPERIMENT_NAME.split("DQN_")[-1].split("_")[0])######## cuando este lo de configuracion 

# Get all the .pt files in the folder 
pt_extension = path + "/*.pt"
pt_files = glob.glob(pt_extension)

pt_files = natsorted(pt_files, key=lambda y: y.lower()) #Sort in natural order


#print("PATH: ", path)
#print("Files: ", pt_files)



epoch_test = []
epoch_CA_intime = []
epoch_CA_late = []
epoch_IA_intime = []
epoch_IA_late = []
epoch_UAC_intime= []
epoch_UAC_late = []
epoch_UAI_intime = []
epoch_UAI_late = []
epoch_CI = []
epoch_II = []
epoch_reward = []
epoch_total_times_execution = []
epoch_total_reward_energy_ep = []
epoch_total_reward_time_ep = []
epoch_total_idle_mean = []
epoch_total_idle = []
epoch_total_idle_var = []
epoch_total_idle_std = []
# epoch_total_time_video = []
epoch_total_time_interaction = []
cont_f  = -1

epoch_G_total = []
epoch_G_energy = []
epoch_G_time = []

cont_actions_test = cfg.ROBOT_CONT_ACTIONS_MEANINGS.copy()
epoch_cont_actions = []
epoch_total_cont_actions_correct= []
cont_actions = 0
prev_decision_rate = 1

if (REACTIVE == True or HEALTHY== True):
    # pdb.set_trace()
    pt_files = pt_files[:2]
    
for f in pt_files:
    print(f)
    epoch = int(f.replace(path + "/model_", '').replace('.pt', ''))
    #print(epoch)
    
    epoch_test.append(epoch)
    
    checkpoint = torch.load(f)
    policy_net.load_state_dict(checkpoint['model_state_dict'])
    policy_net.eval()        
    
    steps_done = 0
    
    total_reward = []
    total_reward_energy = []
    total_reward_time = []
    total_reward_energy_ep = []
    total_reward_time_ep = []
    total_times_execution = []
    
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
    
    total_interaction_time_epoch = []
    #total_maximum_time_execution_epoch = []
    #total_minimum_time_execution_epoch = []
    total_cont_actions_correct = []
    decision_cont = 0
    flag_do_action = True 
    total_idle_mean = []
    total_idle_var = []
    total_idle_std = []
    total_idle =[]
    
    total_G_total = []
    total_G_energy = []
    total_G_time = []
    
    
    for i_episode in range(NUM_EPISODES):

        print("| EPISODE #", i_episode , end='\r')

        state = torch.tensor(env.reset(), dtype=torch.float, device=device).unsqueeze(0)

        done = False
        
        steps_done += 1
        num_optim = 0
        
        G_total = circular_buffer()
        G_energy = circular_buffer()
        G_time= circular_buffer()

        G_total_ep = 0
        G_energy_ep = 0
        G_time_ep = 0
        # action = select_action(state) #1
        
        reward_energy_ep = 0
        reward_time_ep = 0
        if device.type == 'cuda':
            hidden = (torch.randn(1,hidden_size_LSTM).cuda(),
                      torch.randn(1,hidden_size_LSTM).cuda())
        else:
            hidden = (torch.randn(hidden_size_LSTM),
                      torch.randn(hidden_size_LSTM))
        
        for t in count(): 
            decision_cont += 1
            action, flag_decision, prev_decision_rate, hidden = action_rate(decision_cont, state, hidden, prev_decision_rate)
            
            if flag_decision: 
                action_ = action
                action = action.item()
            
            array_action = [action,flag_decision, 'val']
            #next_state_, reward, done, optim, flag_pdb, reward_time, reward_energy, execution_times, correct_action, _ = env.step(array_action)
            next_state_, reward, done, optim, flag_pdb, reward_time, reward_energy, hri_time, correct_action, type_threshold, error_pred, total_pred = env.step([array_action,array_conf])
            
            
            reward = torch.tensor([reward], device=device)
            reward_energy_ep += reward_energy
            reward_time_ep += reward_time

            next_state = torch.tensor([next_state_], dtype=torch.float,device=device)
            
            if flag_decision:
                G_total.append(reward.detach().cpu().numpy())
                G_energy.append(reward_energy)
                G_time.append(reward_time)
                if len(G_total) == N_STEP:
                    for idx_G in range(0,N_STEP): # CUANDO SE GUARDE EN CONFIG CAMBIAR PARA COGER EL STEP DE CONFIG Y NO DEL NOMBRE
                        G_total_ep+= (GAMMA**idx_G)*G_total[idx_G] 
                        G_energy_ep+= (GAMMA**idx_G)*G_energy[idx_G] 
                        G_time_ep+= (GAMMA**idx_G)*G_time[idx_G] 

            # if REACTIVE == True:
            #     if reward_time != 0:
            #         hri_time += 3.5*30
            #         env.modify_time_execution(hri_time)
            #         cont_actions += 1
            if not done: 
                state = next_state
            else:
                next_state = None
            
            if done:
                #total_times_execution.append(execution_times)
                total_reward_energy_ep.append(reward_energy_ep)
                total_reward_time_ep.append(reward_time_ep)
                total_reward.append(env.get_total_reward())
                
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
                total_idle.append(env.idles_list)
                total_idle_mean.append(env.anticipation)  
                total_idle_var.append(env.anticipation_var)  
                total_idle_std.append(env.anticipation_std)
                total_cont_actions_correct.append(list(env.cont_actions_robot.values()))
                
                total_G_total.append(G_total_ep)
                total_G_energy.append(G_energy_ep)
                total_G_time.append(G_time_ep)
                
                #total_time_video = list(list(zip(*total_times_execution))[0])
                #total_time_interaction = list(list(zip(*total_times_execution))[1])
                
                #HRI
                total_interaction_time_epoch.append(hri_time)
                
                #Human baseline
                #total_minimum_time_execution_epoch.append(min_time)
                #total_maximum_time_execution_epoch.append(max_time)                

                
                #print(total_time_video)
                #print(total_time_iteraction)
                
                break #Finish episode

    #epoch_total_times_execution.append(np.sum(total_times_execution))
    epoch_total_reward_energy_ep.append(np.sum(total_reward_energy_ep))
    epoch_total_reward_time_ep.append(np.sum(total_reward_time_ep))

    epoch_G_energy.append(np.sum(total_G_energy))
    epoch_G_time.append(np.sum(total_G_time))
    epoch_G_total.append(np.sum(total_G_total))
    #epoch_total_time_video.append(np.sum(total_time_video))
    epoch_total_time_interaction.append(np.sum(total_interaction_time_epoch)) #HRI time

    epoch_CA_intime.append(np.sum(total_CA_intime))
    epoch_CA_late.append(np.sum(total_CA_late))
    epoch_IA_intime.append(np.sum(total_IA_intime))
    epoch_IA_late.append(np.sum(total_IA_late))
    epoch_UAC_intime.append(np.sum(total_UAC_intime))
    epoch_UAC_late.append(np.sum(total_UAC_late))
    epoch_UAI_intime.append(np.sum(total_UAI_intime))
    epoch_UAI_late.append(np.sum(total_UAI_late))
    epoch_CI.append(np.sum(total_CI))
    epoch_II.append(np.sum(total_II))
    
    # pdb.set_trace()
    epoch_total_idle.append(np.mean(total_idle, axis=0))
    epoch_total_idle_mean.append(np.mean(total_idle_mean))
    epoch_total_idle_var.append(np.mean(total_idle_var))
    epoch_total_idle_std.append(np.mean(total_idle_std))
    epoch_reward.append(np.sum(total_reward))
    epoch_total_cont_actions_correct.append(np.sum(total_cont_actions_correct,axis=0))
    epoch_cont_actions.append(list(cont_actions_test.values()))  


    
    #maximum_time = sum(total_maximum_time_execution_epoch) #Human times
    #minimum_time = sum(total_minimum_time_execution_epoch)
    
    

save_path = os.path.join(path, "Graphics") 
if not os.path.exists(save_path): os.makedirs(save_path)

 

#--------------------ACTIONS ----------------
if (REACTIVE == False and HEALTHY == False):
    fig = plt.figure(figsize=(25, 8))
    plt.subplot(2,5,1)
    plt.title("Short-term correct actions (in time)")
    plt.plot(epoch_test, epoch_CA_intime)
    
    plt.subplot(2,5,5)
    plt.title("Incorrect (in time)")
    plt.plot(epoch_test, epoch_IA_intime)
    
    plt.subplot(2,5,2)
    plt.title("Long-term correct actions (in time)")
    plt.plot(epoch_test, epoch_UAC_intime)
    
    plt.subplot(2,5,4)
    plt.title("Unnecessary incorrect (in time)")
    plt.plot(epoch_test, epoch_UAI_intime)
    
    plt.subplot(2,5,3)
    plt.title("Correct inactions")
    plt.plot(epoch_test, epoch_CI)
    
    plt.subplot(2,5,6)
    plt.title("Short-term correct actions (late)")
    plt.plot(epoch_test, epoch_CA_late)
    plt.xlabel("Epochs")
    
    plt.subplot(2,5,10)
    plt.title("Incorrect (late)")
    plt.plot(epoch_test, epoch_IA_late)
    plt.xlabel("Epochs")
    
    plt.subplot(2,5,7)
    plt.title("Long-term correct actions (late)")
    plt.plot(epoch_test, epoch_UAC_late)
    plt.xlabel("Epochs")
    
    plt.subplot(2,5,9)
    plt.title("Unnecessary incorrect (late)")
    plt.plot(epoch_test, epoch_UAI_late)
    plt.xlabel("Epochs")
    
    plt.subplot(2,5,8)
    plt.title("Incorrect inactions")
    plt.plot(epoch_test, epoch_II)
    plt.xlabel("Epochs")

    if PERSONALIZATION:
            fig.savefig(save_path+'/00_TEST__PERSONALIZATION_INTERACTION_TIME_'+TYPE_PERSONALIZATION+'.jpg')
    else:
        
        if env.test: fig.savefig(save_path+'/00_TEST_ACTIONS.jpg')
        else: fig.savefig(save_path+'/00_TRAIN_ACTIONS.jpg')
        





 # ---------------------------------------------------------------------------------------------

# ------------------- ACTIONS ------ PIE CHART
# (ONLY IN THE LAST EPOCH)
best_time_position = epoch_total_time_interaction.index(min(epoch_total_time_interaction))

# # -------------__REWARDS -------------------------

if LOOP == False:
    NUM_LOOP = str(0)
else:
    NUM_LOOP = str(NUM_LOOP)
    
if (REACTIVE == False and HEALTHY==False):
    
    fig2 = plt.figure(figsize=(20,6))
    plt.subplot2grid((1,3),(0,0))
    
    plt.plot(epoch_total_reward_energy_ep, 'c:')
    plt.title("Energy reward")
    plt.legend(["Energy reward"])
    plt.xlabel("Epoch")
    plt.subplot2grid((1,3),(0,1))
    
    plt.plot(epoch_total_reward_time_ep, 'c:')
    plt.title("Time reward")
    plt.legend(["Time reward"])
    plt.xlabel("Epoch")
    
    plt.subplot2grid((1,3),(0,2))
    
    plt.title("Total reward")
    plt.plot(epoch_reward, 'c-.')
    plt.legend(["Total reward"])
    plt.xlabel("Epoch")
    
    # plt.show()
    if PERSONALIZATION:
            fig2.savefig(save_path+'/00_TEST__PERSONALIZATION_INTERACTION_TIME_'+TYPE_PERSONALIZATION+'_iter_'+NUM_LOOP+'.jpg')
    else:
        if env.test: fig2.savefig(save_path+'/00_TEST_REWARD'+'_iter_'+NUM_LOOP+'.jpg')
        else: fig2.savefig(save_path+'/00_TRAIN_REWARD.jpg')
#-----------------------------
#---------- discounted REWARDs ---------------------------
fig1 = plt.figure(figsize=(20, 6))
 
plt.subplot2grid((1,3), (0,0))
plt.plot(epoch_G_energy, 'b:')
plt.legend(["Energy discounted reward"])
plt.xlabel("Epoch")

plt.title(" Test | Energy discounted reward")

plt.subplot2grid((1,3), (0,1))
plt.plot(epoch_G_time, 'b:')
plt.legend(["Time discounted reward"])
plt.xlabel("Epoch")

plt.title(" Test | Time discounted reward")


plt.subplot2grid((1,3), (0,2))
plt.plot(epoch_G_total, 'b-.')
plt.xlabel("Epoch")
plt.legend(["Total discounted reward"])
plt.title("Total discounted reward (G)")

fig1.savefig(save_path+'/00_DISCOUNTED_REWARD_iter_'+NUM_LOOP+'.jpg')
plt.close(fig1)

plt.close('all')
epoch_total_time_interaction[:] = [x / 30 for x in epoch_total_time_interaction]

#--------------- INTERACTION ---------------------
if (REACTIVE == False and HEALTHY == False):
    fig3 = plt.figure(figsize=(10,6))
    plt.title("Interaction time")
    # plt.plot(epoch_total_time_video, 'k',label='Video')
    plt.axhline(y=healthy_human_time/30, color='k', label='Healthy human')
    plt.plot(epoch_total_time_interaction, 'c--',label='Interaction')
    plt.axhline(y=minimum_time/30, color='r', label='Minimum')
    plt.legend()
    # plt.xlabel('Failure rate')
    # plt.xticks([0,1,2], labels=["0","0.5","0.8"])
    plt.ylabel("Time (s)")
    
    # plt.show()
    if PERSONALIZATION:
        fig3.savefig(save_path+'/00_TEST__PERSONALIZATION_INTERACTION_TIME_'+TYPE_PERSONALIZATION+'_iter_'+NUM_LOOP+'.jpg')
    else:
        if env.test: fig3.savefig(save_path+'/00_TEST_INTERACTION_TIME_iter_'+NUM_LOOP+'.jpg')
        else: fig3.savefig(save_path+'/00_TRAIN_INTERACTION_TIME.jpg')


file_name = pt_files[best_time_position]

if REACTIVE== True:
    with open(path +'/best_epoch_test_iter_time_reactive_iter_'+NUM_LOOP+'.txt', 'w') as f: f.write(file_name.split('/')[-1])
elif PERSONALIZATION == True:
    with open(path +'/best_epoch_test_iter_personalization_time_iter_'+NUM_LOOP+'.txt', 'w') as f: f.write(file_name.split('/')[-1])

else:
    with open(path +'/best_epoch_test_iter_time_iter_'+NUM_LOOP+'.txt', 'w') as f: f.write(file_name.split('/')[-1])

data_test = {
'Short-term_CA': epoch_CA_intime,
'Short_term_CA_late':epoch_CA_late,
'IA_intime': epoch_IA_intime,
'IA_late':epoch_IA_late,
'Long-term_CA': epoch_UAC_intime,
'Long-term_CA_late': epoch_UAC_late,
'UAI_intime': epoch_UAI_intime,
'UAI_late': epoch_UAI_late,
'CI': epoch_CI,
'II': epoch_II,
'mean_idle': epoch_total_idle_mean,
'var_idle': epoch_total_idle_var,
'std_idle': epoch_total_idle_std,
'idle':epoch_total_idle,
'time_interaction': epoch_total_time_interaction,
'energy_reward': epoch_total_reward_energy_ep,
'time_rreward':epoch_total_reward_time_ep,
'G_total': epoch_G_total,
'G_energy': epoch_G_energy,
'G_time': epoch_G_time,
'minimum_time': minimum_time/30,
'healthy_human_time': healthy_human_time/30,
'best_epoch': file_name.split('/')[-1]
}

df_test = pd.DataFrame(data_test)
if REACTIVE == True:
    data_test = {
        'reactive_time': epoch_total_time_interaction,
        'energy_reward': epoch_total_reward_energy_ep
        }
    df_test = pd.DataFrame(data_test)
    df_test.to_csv(save_path+'/results_test_REACTIVE.csv')
    print('REACTIVE: ',epoch_total_time_interaction[0])
    print('TOTAL ACTIONS TO BE DONE: ',cont_actions)
elif HEALTHY == True:
    data_test = {
        'healthy_time': epoch_total_time_interaction,
        'energy_reward': epoch_total_reward_energy_ep
        }
    df_test = pd.DataFrame(data_test)
    df_test.to_csv(save_path+'/results_test_REACTIVE_.csv')
    print('HEALTHY: ',epoch_total_time_interaction[0])
    
        
else:
    if PERSONALIZATION:
        # AÑADIR LOS VALORES NUEVOS DE PERSON
        df_test['epoch_cont_actions'] = epoch_cont_actions
        df_test['epoch_total_cont_actions_correct'] = epoch_total_cont_actions_correct
        df_test.to_csv(save_path+'/results_test_PERSONALIZATION_'+TYPE_PERSONALIZATION+'_iter_'+NUM_LOOP+'.csv')
    else:
        data_idle ={'idle':epoch_total_idle}
        df_idle = pd.DataFrame(data_idle)
        df_idle.to_csv(save_path+'/results_idle_iter_'+NUM_LOOP+'.csv')
        df_test.to_csv(save_path+'/results_test_iter_'+NUM_LOOP+'.csv')
        
        
