import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random
import time
from aux import *
import config as cfg
import numpy as np
import glob
import pdb
from numpy import random
import pandas as pd
import random
from collections import Counter
import copy
from natsort import natsorted, ns
import pickle

#CONFIGURATION GLOBAL ENVIRONMENT VARIABLES
ACTION_SPACE = cfg.ACTION_SPACE
N_ATOMIC_ACTIONS = cfg.N_ATOMIC_ACTIONS
N_OBJECTS = cfg.N_OBJECTS
VWM = (N_OBJECTS-1)*2

ATOMIC_ACTIONS_MEANINGS = cfg.ATOMIC_ACTIONS_MEANINGS
OBJECTS_MEANINGS = cfg.OBJECTS_MEANINGS
ROBOT_ACTIONS_MEANINGS = copy.deepcopy(cfg.ROBOT_ACTIONS_MEANINGS)
# ROBOT_ACTION_DURATIONS = cfg.ROBOT_ACTION_DURATIONS
ROBOT_POSSIBLE_INIT_ACTIONS = cfg.ROBOT_POSSIBLE_INIT_ACTIONS
OBJECTS_INIT_STATE = copy.deepcopy(cfg.OBJECTS_INIT_STATE)

VERSION = cfg.VERSION
POSITIVE_REWARD = cfg.POSITIVE_REWARD

Z_hidden_state = cfg.Z_hidden_state
Z_HIDDEN = cfg.Z_HIDDEN

INTERACTIVE_OBJECTS_ROBOT = copy.deepcopy(cfg.INTERACTIVE_OBJECTS_ROBOT)
ONLY_RECOGNITION = cfg.ONLY_RECOGNITION
NORMALIZE_DATA = cfg.NORMALIZE_DATA

#ANNOTATION-RELATED VARIABLES

if ONLY_RECOGNITION:
    dataset = 'dataset_pred'
else:
    dataset = 'dataset_pred_recog_tmp_ctx'
    
root_realData = "./video_annotations/"+dataset+"/*" #!
videos_realData = glob.glob(root_realData) #Folders

if cfg.PERSONALIZATION:
    if cfg.TYPE_PERSONALIZATION == '':
        with open("./video_annotations/"+cfg.PERSON+"_test.txt") as f:
            lines_x = f.readlines()
            lines = []
            for line in lines_x:
                lines.append(root_realData.split('*')[0]+line.split('\n')[0])
        videos_realData_test=lines
        # print(lines)
        train_videos = list(set(videos_realData)-set(videos_realData_test))
        videos_realData = train_videos
    else:
        with open("./video_annotations/"+cfg.PERSON+"_test.txt") as f:
            lines_x = f.readlines()
            lines = []
            for line in lines_x:
                lines.append(root_realData.split('*')[0]+line.split('\n')[0])
        videos_realData_test=lines
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


random.shuffle(videos_realData)
total_videos = len(videos_realData)

video_idx = 0 #Index of current video
action_idx = 0 #Index of next_action
frame = 0 #Current frame
recipe = '' #12345

correct_action = -1 # esto para que es
labels_pkl = 'labels_updated.pkl'
path_labels_pkl = os.path.join(videos_realData[video_idx], labels_pkl)

annotations = np.load(path_labels_pkl, allow_pickle=True)


class BasicEnv(gym.Env):
    message = "Custom environment for recipe preparation scenario."


    def __init__(self, display=False, test=False):
        self.action_space = gym.spaces.Discrete(ACTION_SPACE) #[0, ACTION_SPACE-1]

        if ONLY_RECOGNITION == True:
            if Z_hidden_state:
                self.observation_space = gym.spaces.Discrete(N_ATOMIC_ACTIONS+VWM+N_OBJECTS+Z_HIDDEN) # Next Action + Action Recog + VWM + Obj in table + Z
            else:
                self.observation_space = gym.spaces.Discrete(N_ATOMIC_ACTIONS+VWM+N_OBJECTS)
        else:
            if Z_hidden_state:
                self.observation_space = gym.spaces.Discrete(N_ATOMIC_ACTIONS*2+VWM+N_OBJECTS+Z_HIDDEN) # Next Action + Action Recog + VWM + Obj in table + Z
            else:
                self.observation_space = gym.spaces.Discrete(N_ATOMIC_ACTIONS*2+VWM+N_OBJECTS)

        self.state = [] #One hot encoded state
        self.total_reward = 0
        #self.prev_state = []
        self.action_repertoire = ROBOT_ACTIONS_MEANINGS
        self.next_atomic_action_repertoire = ATOMIC_ACTIONS_MEANINGS
        self.display = display
        self.test = test
        self.flags = {'freeze state': False, 'decision': False, 'threshold': " ",'evaluation': "Not evaluated", 'action robot': False,'break':False,'pdb': False}
        self.person_state = "other manipulation"
        self.robot_state = "Predicting..."
        self.reward_energy = 0
        self.reward_time = 0
        self.time_execution = 0
        self.mode = 'train'
        self.objects_in_table = OBJECTS_INIT_STATE.copy()
        self.df_actions_to_be_done_robot_video = pd.DataFrame() 
        
        global root, videos_realData, total_videos, annotations

        if self.test:
            print("==== TEST SET ====")
            # random.seed(0)
            total_videos = len(videos_realData_test)
            labels_pkl = 'labels_updated.pkl'
            videos_realData = videos_realData_test
            path_labels_pkl = os.path.join(videos_realData[video_idx], labels_pkl)
            annotations = np.load(path_labels_pkl, allow_pickle=True)

        self.CA_intime = 0
        self.CA_late = 0
        self.IA_intime = 0
        self.IA_late = 0
        self.UAC_intime = 0
        self.UAC_late = 0
        self.UAI_intime = 0
        self.UAI_late = 0
        self.CI = 0
        self.II = 0
        self.cont_actions_robot = cfg.ROBOT_CONT_ACTIONS_MEANINGS.copy()
        #12345
        self.UA_related = 0
        self.UA_unrelated = 0
        self.prediction_error = 0
        self.total_prediction = 0

        self.r_history = []
        self.h_history = []
        self.rwd_history = []
        self.rwd_time_h = []
        self.rwd_energy_h = []
        
        self.anticipation = 0
        self.anticipation_var = 0
        self.anticipation_std = 0
        self.idles = 0
        self.idles_list = []
        
        
    def get_frame(self):
        global frame
        return frame
    def get_action_idx(self):
        global action_idx
        return action_idx
    def get_annotations(self):
        global annotations_orig
        return annotations
    def get_action_meanings(self):
        return self.action_repertoire
    def get_state_meanings(self):
        return self.state_repertoire

    def modify_time_execution(self, value):
        self.time_execution = value
    def energy_robot_reward (self, action):
        global FACTOR_ENERGY_PENALTY
        self.reward_energy = -cfg.ROBOT_AVERAGE_DURATIONS[action]*FACTOR_ENERGY_PENALTY #ENERGY PENALTY

    def get_possibility_objects_in_table (self, annot):

        person_states = annot['label']

        objects_video = []
        in_table_video = []
        fr_init_video = []
        fr_end_video = []
        index_annotation = []

        for idx,value in enumerate(person_states):
            for obj in INTERACTIVE_OBJECTS_ROBOT:
                if obj in ATOMIC_ACTIONS_MEANINGS[value]:
                    if 'extract' in ATOMIC_ACTIONS_MEANINGS[value]:
                        objects_video.append(obj)
                        in_table_video.append(1)
                        fr_init_video.append(0)
                        fr_end_video.append(annot['frame_init'][idx])
                        index_annotation.append(idx)

        video_bring = {"Object": objects_video, "In table": in_table_video, "Frame init": fr_init_video,"Frame end": fr_end_video, "Index": index_annotation}

        res = all(ele == [] for ele in list(video_bring.values())) #check if video bring is empty
        if res == False:
            df_bring = pd.DataFrame(video_bring)
            # tengo que cerar
            for idx,value in enumerate(person_states):
                for obj in INTERACTIVE_OBJECTS_ROBOT:
                    if obj in ATOMIC_ACTIONS_MEANINGS[value]:
                        if 'put' in ATOMIC_ACTIONS_MEANINGS[value]: ##### esto habria que cambiarlo cuando se le vaya a incorporar put, que no dependa de que haya accion bring

                            df_current_object=df_bring[df_bring["Object"] == obj]
                            # este caso es distinto
                            if not df_current_object.empty:
                                objects_video.append(obj)
                                in_table_video.append(0)
                                fr_init_video.append(annot['frame_end'][int(df_current_object['Index'])])
                                fr_end_video.append(annot['frame_init'][idx])

            data_video =  {"Object": objects_video, "In table": in_table_video, "Frame init": fr_init_video,"Frame end": fr_end_video}
            df_video = pd.DataFrame(data_video)
            person_states_print = []
            for idx,val in enumerate(person_states):
                person_states_print.append(ATOMIC_ACTIONS_MEANINGS[val])
        else:
            df_video = pd.DataFrame()

        return df_video

    def get_minimum_execution_times(self):

        global annotations
        df_video = self.get_possibility_objects_in_table(annotations)
        name_actions = []
        total_minimum_time_execution = annotations['frame_end'][len(annotations)-1]
        df_video_dataset = df_video.copy()
        if not df_video.empty:
            for index, row in df_video.iterrows():
                if row['In table'] == 1:
                    name_actions.append("bring "+row['Object'])
                else:
                    name_actions.append("put "+row['Object']+ ' fridge')
    
            df_video['Actions'] = name_actions
            keys = list(df_video['Actions'])
            video_dict = {}
            for i in keys:
                video_dict[i] = 0
    
            person_states = annotations['label']
            
            # total_minimum_time_execution = 0
           
            # df_video_dataset = df_video
            df_video_dataset = df_video.copy()
            df_video_dataset['Max_time_to_save'] = [0]*len(df_video_dataset)
            df_video_dataset['action_idx'] = [0]*len(df_video_dataset)
    
            if not df_video_dataset.empty:
                for idx,value in enumerate(person_states):
                    for obj in INTERACTIVE_OBJECTS_ROBOT:
                        if obj in ATOMIC_ACTIONS_MEANINGS[value]:
                            if 'extract' in ATOMIC_ACTIONS_MEANINGS[value]:
                                if idx != 0:
                                    action_name = 'bring '+ obj
                                    fr_init = annotations['frame_init'][idx-1]
    
                                    df_video_dataset['Frame init'].loc[df_video_dataset['Actions']==action_name] = fr_init
                                    keys = [k for k, v in ROBOT_ACTIONS_MEANINGS.items() if v == action_name]
                                    df_video_dataset['Max_time_to_save'].loc[df_video_dataset['Actions']==action_name] = annotations['frame_end'][idx] - annotations['frame_end'][idx-1]
                                    df_video_dataset['action_idx'].loc[df_video_dataset['Actions']==action_name] = idx
                            # elif 'put' in ATOMIC_ACTIONS_MEANINGS[value]:
                            #     action_name = 'put '+ obj +' fridge'
                            #     fr_init = annotations['frame_init'][idx-1]
                            #     # df_selected = df_video_dataset.loc[df_video_dataset['Actions']==action_name]
                            #     df_video_dataset['Frame init'].loc[df_video_dataset['Actions']==action_name] = fr_init
                            #     keys = [k for k, v in ROBOT_ACTIONS_MEANINGS.items() if v == action_name]
    
                            #     df_video_dataset['Max_time_to_save'].loc[df_video_dataset['Actions']==action_name] = annotations['frame_end'][idx] - annotations['frame_end'][idx-1]
                            #     # df_video_dataset['Max_time_to_save'].loc[df_video_dataset['Actions']==action_name] = annotations['frame_end'][idx] - annotations['frame_end'][idx-1] - ROBOT_ACTION_DURATIONS[keys[0]]
                            #     df_video_dataset['action_idx'].loc[df_video_dataset['Actions']==action_name] = idx
    
    
                df_video_dataset.sort_values("Frame init")
                # print(df_video_dataset)
                total_minimum_time_execution = annotations['frame_end'][len(annotations)-1] - df_video_dataset['Max_time_to_save'].sum()
        return total_minimum_time_execution,df_video_dataset

    def get_energy_robot_reward(self,action):
        global memory_objects_in_table, frame, annotations_orig

        df_video = self.get_possibility_objects_in_table(annotations_orig)
        energy_reward = 0

        if not df_video.empty:
            variations_in_table = len(memory_objects_in_table)
            if variations_in_table < 2:
                oit_prev = memory_objects_in_table[0]
                oit = memory_objects_in_table[0]
            else:
                oit_prev = memory_objects_in_table[variations_in_table-2]
                oit = memory_objects_in_table[variations_in_table-1]
    
            objects_prev_print = []
            for key,value in OBJECTS_MEANINGS.items():
                if oit_prev[key] == 1:
                    objects_prev_print.append(value)
    
            objects_print = []
            for key,value in OBJECTS_MEANINGS.items():
                if oit[key] == 1:
                    objects_print.append(value)
    
            set1 = set(objects_prev_print)
            set2 = set(objects_print)
    
            missing = list(sorted(set1 - set2))
            added = list(sorted(set2 - set1))
    
            if len(missing) > 0:
                # print("PUT ")
                if (df_video['Object'] == missing[0]).any():
                    df = df_video.loc[df_video['Object'] == missing[0]]
                    if (df['In table'] == 0).any():
                        df_ = df.loc[df['In table'] == 0]
                        fr_init = int(df_['Frame init'])
                        fr_end = int(df_['Frame end'])
                        if fr_init < frame < fr_end:
                            energy_reward = 0
                        else:
                            energy_reward = 1
                    else:
                        energy_reward = 1
                else:
                     energy_reward = 1
    
            elif len(added) > 0:
    
                # print("BRING ")
                if (df_video['Object'] == added[0]).any():
                    df = df_video.loc[df_video['Object'] == added[0]]
                    if (df['In table'] == 1).any():
                        df_ = df.loc[df['In table'] == 1]
                        fr_init = int(df_['Frame init'])
                        fr_end = int(df_['Frame end'])
    
                        if fr_init < frame < fr_end:
                            energy_reward = 0
                        else:
                            energy_reward = 1
                    else:
                        energy_reward = 1
                else:
                     energy_reward = 1
    
            self.energy_robot_reward(action)
            self.reward_energy = energy_reward * self.reward_energy
        else: # if empty means thata all actions are unnecesary
            self.reward_energy = self.reward_energy
            
    def update_objects_in_table (self, action):

        meaning_action = ROBOT_ACTIONS_MEANINGS.copy()[action]
        for obj in INTERACTIVE_OBJECTS_ROBOT:
            if obj in meaning_action:

                if 'bring' in meaning_action:
                    self.objects_in_table[obj] = 1
                elif 'put' in meaning_action:
                    self.objects_in_table[obj] = 0

    def possible_actions_taken_robot (self):
        bring_actions = []
        for x in INTERACTIVE_OBJECTS_ROBOT:
            bring_actions.append(''.join('bring '+ x))

        put_actions = []
        for x in INTERACTIVE_OBJECTS_ROBOT:
            put_actions.append(''.join('put '+ x + ' fridge'))

        position_bring_actions = [ele for ele in [key for key,value in ROBOT_ACTIONS_MEANINGS.copy().items() if value in bring_actions]]
        order_objects_bring = [ele for ele in [value for key,value in ROBOT_ACTIONS_MEANINGS.copy().items() if value in bring_actions]]
        position_put_actions = [ele for ele in [key for key,value in ROBOT_ACTIONS_MEANINGS.copy().items() if value in put_actions]]
        order_objects_put = [ele for ele in [value for key,value in ROBOT_ACTIONS_MEANINGS.copy().items() if value in put_actions]]

        dict_bring = {}
        for idx,value in enumerate(order_objects_bring):
            for obj in INTERACTIVE_OBJECTS_ROBOT:
                if obj in value:
                    dict_bring[obj] = position_bring_actions[idx]

        dict_put = {}
        for idx,value in enumerate(order_objects_put):
            for obj in INTERACTIVE_OBJECTS_ROBOT:
                if obj in value:
                    dict_put[obj] = position_put_actions[idx]

        # dict_put = {objects[i]: position_put_actions[i] for i in range(len(objects))}
        possible_actions = [0]*len(ROBOT_ACTIONS_MEANINGS)

        for key,value in ROBOT_ACTIONS_MEANINGS.copy().items():
            for obj in INTERACTIVE_OBJECTS_ROBOT:
                if obj in value:
                    if self.objects_in_table[obj] == 0:
                        idx = dict_bring.copy()[obj]
                        possible_actions[idx] = 1
                    else:
                        try:
                            idx_put = dict_put.copy()[obj]
                            possible_actions[idx_put] = 1
                        except: pass

        possible_actions[5] = 1

        return possible_actions

    def select_inaction_sample (self, inaction):
        # random_position = random.randint(0,len(inaction)-1)
        random_position = len(inaction)-1 # AHORA CON N-STEP, NOS QUEDAMOS CON EL ULTIMO, YA QUE GUARDAMOS DE VEZ EN CUANDO NO ACCIONES EN LA MEMORIA
        self.state = inaction[random_position][1] # esto se hace para que el next_state sea el siguiente al guardado
        reward = inaction[random_position][2]

        return reward

    def select_correct_action (self, action):

        global frame, annotations, ROBOT_ACTION_DURATIONS, healthy_human

        length = len(annotations['label']) -1
        last_frame = int(annotations['frame_end'][length])

        for idx, val in ROBOT_ACTION_DURATIONS.items():
            reward = self._take_action(idx)
            if reward > -1 and idx!=5:
                correct_action = idx
                duration_action = val

        # print("Acción tomada: ",cfg.ROBOT_ACTIONS_MEANINGS[action])
        # print("Corrección de accion: ",cfg.ROBOT_ACTIONS_MEANINGS[correct_action])
        if healthy_human==False:
            self.time_execution += 3.5*30 # CUANDO SE CONFUNDE SE AÑADE ESTE TIEMPO DE ASR
        new_threshold = duration_action + frame
        if new_threshold > last_frame:
            new_threshold = last_frame

        return new_threshold, correct_action

    def select_correct_action_video (self, action_idx):

        global frame, annotations, ROBOT_ACTION_DURATIONS

        length = len(annotations['label']) -1
        last_frame = int(annotations['frame_end'][length])

        real_state = annotations['label'][action_idx]

        for idx, val in ROBOT_ACTION_DURATIONS.items():
            reward = self._take_action(idx, real_state)
            if reward > -1:
                correct_action = idx
                duration_action = val

        new_threshold = duration_action + frame
        if new_threshold > last_frame:
            new_threshold = last_frame

        return new_threshold, correct_action

    def update(self, update_type, action):
        global frame, action_idx, inaction, annotations

        length = len(annotations['label']) - 1
        fr_init_next = int(annotations['frame_init'][action_idx])
        fr_end = int(annotations['frame_end'][action_idx-1])

        if self.flags['threshold'] == "last":
            self.flags['break'] = True
        else:
            if update_type == "action":
                df_video = self.df_actions_to_be_done_robot_video.copy()
                if not df_video.empty:
                    df_video_filtered = df_video[df_video['Actions']==ROBOT_ACTIONS_MEANINGS[action]]
                    if df_video_filtered.empty:
                        frame = int(annotations['frame_end'][action_idx])-1
                        if action_idx + 1 <= length:
                            action_idx = action_idx + 1
                    else:
                        if action_idx != df_video_filtered['action_idx'].item(): # se hace na accion antes de tiempo
                        # como ya se hizo la accion se quita de las annotaciones del video, la persona ya no va a hacerlo
                        # se tiene que quitar teniendo en cuenta los tiempos correspondientes
                            drop_future = annotations.iloc[df_video_filtered['action_idx'].item()]
                            time_saving = drop_future['frame_end'] - annotations.iloc[df_video_filtered['action_idx'].item()-1]['frame_end']
                            # print('pre annotations: \n', annotations)
                            # print(annotations)
                            for idx, row in annotations.iterrows():
                                if idx > df_video_filtered['action_idx'].item():
                                    annotations['frame_init'].at[idx] = (annotations['frame_init'].at[idx] - time_saving)
                                    annotations['frame_end'].at[idx] = (annotations['frame_end'].at[idx] - time_saving)
                            annotations_post = annotations.drop(df_video_filtered['action_idx'].item())
                            annotations = annotations_post.reset_index(drop=True)
                            # print(annotations)
                            # pdb.set_trace()
                            _,self.df_actions_to_be_done_robot_video = self.get_minimum_execution_times() # esto se tiene que actualizar, action_idx puede cambiar
                            if not self.df_actions_to_be_done_robot_video.empty:
                                self.df_actions_to_be_done_robot_video = self.df_actions_to_be_done_robot_video[self.df_actions_to_be_done_robot_video['In table']==1]
                            # print('post annotations: \n', annotations)
                            
                            if self.flags['threshold'] == 'parada':
                                # print('Frame end: ', fr_end)
                                # print('Frame init next: ', fr_init_next)
                                if frame > fr_end:
                                    frame = fr_end
                            if self.flags['threshold'] == ('second' or 'next action init'):
                                frame = fr_init_next
                                if action_idx + 1 <= length:
                                    action_idx = action_idx + 1
                            elif self.flags['threshold'] == ('first' or 'next action'):
                                if frame > fr_end:
                                    frame = fr_end
                            # if action == 1:
                            #     print(self.flags)
                            #     print('Frame end: ', fr_end)
                            #     print('Frame init next: ', fr_init_next)
                            #     pdb.set_trace()

                        else:   
                            frame = int(annotations['frame_end'][action_idx])
                            if action_idx + 1 <= length:
                                action_idx = action_idx + 1
                else:
                    frame = int(annotations['frame_end'][action_idx])
                    if action_idx + 1 <= length:
                        action_idx = action_idx + 1
                inaction = []

            elif update_type == "unnecesary":
                if self.flags['threshold'] == ('second' or 'next action init'):
                    frame = fr_init_next
                    if action_idx + 1 <= length:
                        action_idx = action_idx + 1
                elif self.flags['threshold'] == ('first' or 'next action'):
                    if frame > fr_end:
                        frame = fr_end
                inaction = []

    def time_course (self, action):
        global frame, action_idx, inaction, ROBOT_ACTION_DURATIONS,ERROR_PROB

        sample = random.random() #0000

        if (action==5) and (self.flags['decision']):
            self.idles += 1
        elif (action != 5) and (self.flags['decision']):
            # print("Idles before action: ", self.idles)
            self.idles_list.append(self.idles)
            self.idles = 0
            
        if sample < ERROR_PROB:
            self.duration_action = int(random.gauss(1.5*ROBOT_ACTION_DURATIONS[int(action)], 0.2*ROBOT_ACTION_DURATIONS[int(action)]))
            fr_execution = self.duration_action + frame
        else:
            self.duration_action = int(random.gauss(ROBOT_ACTION_DURATIONS[int(action)], 0.2*ROBOT_ACTION_DURATIONS[int(action)]))
            fr_execution = self.duration_action + frame

        # if action!=5:
            # pdb.set_trace()
        fr_end = int(annotations['frame_end'][action_idx-1])
        fr_init_next = int(annotations['frame_init'][action_idx])
        last_frame = int(annotations['frame_end'].iloc[-1])
        # fr_end_next = int(annotations['frame_end'][action_idx])
        # df_video_filtered = pd.DataFrame()

        if self.flags['decision']:
            self.flags['freeze state'] = True
        else:
            self.flags['freeze state'] = False

        self.flags['threshold'] = ''

        if action !=5:
            # if the execution of the action takes more time than the video itself, the execution time of the action == last frame video
            if fr_execution > last_frame:
                threshold = last_frame
                fr_execution = last_frame
            # if time execution is less than the end of the current action, the evaluation time is when the person finish its action
            if fr_execution < fr_end:
                    threshold = fr_end
                    self.flags['threshold'] = "first"
            else:
                df_video_filtered = self.df_actions_to_be_done_robot_video.copy()
                increase_threshold = True
                # VA A HABER QUE HACER UNA ACCION?
                for index, row in df_video_filtered.iterrows():
                   if row['Frame init'] <= frame < row['Frame end']:
                       if self.objects_in_table[row['Object']] != row['In table']:  # nunca llega a hacer la 1 accion, que no se pare
                            if row['Frame end'] != int(annotations['frame_init'][0]):
                               increase_threshold = False

                   if row['Frame init'] <= fr_execution < row['Frame end']: # esto no quiere decir que sea inmediatamente la siguiente, la que tiene que hacerse
                        if self.objects_in_table[row['Object']] != row['In table']: # nunca llega a hacer la 1 accion, que no se pare
                            if row['Frame end'] != int(annotations['frame_init'][0]):
                                increase_threshold = False
                                
                   if fr_execution > row['Frame end']: # esto no quiere decir que sea inmediatamente la siguiente, la que tiene que hacerse
                        if self.objects_in_table[row['Object']] != row['In table']: # nunca llega a hacer la 1 accion, que no se pare
                            if row['Frame end'] != int(annotations['frame_init'][0]):
                                increase_threshold = False

                        
                # se puede ampliar el thr si fuera necesario (la siguiente accion no la tiene que hacer el robot)
                if increase_threshold == True:
                    if fr_execution < fr_init_next:
                        threshold = fr_init_next
                        fr_end = fr_init_next
                        self.flags['threshold'] = "second"
                    else:
                        self.flags['freeze state'] = False
                        threshold = int(annotations['frame_end'][action_idx])
                        self.flags['threshold'] = "next action"

                        if fr_execution > threshold: # si sigue siendo mayor
                            if action_idx + 1 <= len(annotations['label']) - 1:
                                threshold = int(annotations['frame_init'][action_idx+1])
                                self.flags['threshold'] = "next action init"

                        if action_idx + 1 <= len(annotations['label']) - 1: # si sigue siendo mayor 
                            if fr_execution > int(annotations['frame_init'][action_idx+1]):
                                threshold = int(annotations['frame_end'][action_idx+1])
                                self.flags['threshold'] = "next action"

                        fr_end = threshold
                        if fr_execution > last_frame or action_idx == len(annotations['label']) - 1:
                            # print("AL FINAL NO")
                            self.flags['freeze state'] = True
                            threshold = last_frame
                            fr_execution = last_frame
                            fr_end = last_frame
                            self.flags['threshold'] = "last"

                # si increase threshold == False quiere decir que estamos en momento de hacer algo, el th no se incrementa mas alla
                else:
                    # print("NO SE INCREMENTA")
                    # self.flags['freeze state'] = True
                    if frame < fr_end:
                        threshold = max(fr_execution, fr_end)
                        self.flags['threshold'] = "first"
                    else:
                        threshold = max(fr_execution, fr_init_next)
                        fr_end = fr_init_next
                        self.flags['threshold'] = "second"
                                
        else:
            if frame == fr_end - 1 or frame == fr_init_next - 1:
                if len(inaction) > 0:
                    if "action" not in inaction:
                        # flag_decision = True
                        self.flags['decision'] = True
                        # self.flags['freeze state'] = True
                if frame == fr_init_next - 1:
                    threshold = fr_init_next
                    fr_end = fr_init_next
                    self.flags['threshold'] = "second"
                else:
                    threshold = fr_end
                    self.flags['threshold'] = "first"
            else:
                threshold = frame

        return threshold, fr_execution, fr_end

    def evaluation(self, action, fr_execution, fr_end, frame_post):
        global frame, action_idx, inaction, new_energy, correct_action, recipe

        optim = True
        simple_reward = self._take_action(action)
        new_threshold = 0
        reward = 0

        if (self.flags['evaluation'] == 'Incorrect action' or self.flags['evaluation'] == "Incorrect inaction"):
            if frame == fr_execution:
                self.flags['evaluation'] = 'Not evaluated'
                frame_post.append(frame)
                reward = self.reward_energy + self.reward_time
                self.update("action",action)
                self.flags['break'] = True

            else:
                self.reward_time += -1
        else:
            if simple_reward == 5:
                if self.flags['threshold'] == '':
                    optim = False
                elif frame == fr_end:
                    # CORRECT INACTION
                    if simple_reward > -1:
                        self.cont_actions_robot[action] += 1
                        # no se actualiza como antes
                        self.CI += 1
                        # print("Action idx: ", action_idx)
                        # print("*************** CORRECT INACTION ***************")
                        reward = self.select_inaction_sample(inaction)
                
                
            # se hace otra accion
            elif simple_reward == -5:
                if fr_execution <= fr_end:
                    if frame == fr_execution:
                        frame_post.append(frame)
                    if frame == fr_end:
                        inaction.append("action")
                        self.energy_robot_reward(action)
                        self.get_energy_robot_reward(action)
                        reward = self.reward_energy
                        if reward == 0: # action done very anticipated
                            self.UAC_intime += 1
                            self.cont_actions_robot[action] += 1
                            self.update("action",action)
                            # print("*************** LONG-TERM CORRECT ACTION ***************")
                        else:
                            self.UAI_intime += 1
                            if recipe == 'c' or recipe == 'd':
                                if action == 2: self.UA_related += 1
                                else: self.UA_unrelated += 1
                            elif recipe == 't':
                                if action in [0, 1, 3, 4]: self.UA_related += 1
                                else: self.UA_unrelated += 1
                        # self.update("action",action) #################### NO
                            self.update("unnecesary",action)
                else:
                    if frame > fr_end:
                        self.reward_time += -1
                    if frame == fr_execution:
                        # print("Action idx: ", action_idx)
                        
                        inaction.append("action")
                        frame_post.append(frame)
                        self.energy_robot_reward(action)
                        self.get_energy_robot_reward(action)
                        reward =  self.reward_energy + self.reward_time
                        if self.reward_energy == 0: # action done very anticipated
                            self.UAC_late += 1
                            self.update("action",action)
                            # print("*************** LONG-TERM CORRECT ACTION (late) ***************")
                        else:
                            self.UAI_late += 1
                            # print("*************** UNNECESARY ACTION (late) ***************")
                            # pdb.set_trace()
                        # self.update("action",action) #################### NO
                        self.update("unnecesary",action)
                        self.flags['break'] = True
                # else:
                #     print("QUE ES ESTO ?¿?¿?")
                #     pdb.set_trace()
            else:
                if action !=5:
                    # CORRECT ACTION
                    if simple_reward > -1:
                        # In time
                        if fr_execution <= fr_end:
                            if frame == fr_execution:
                                frame_post.append(frame)
                                self.reward_energy = 0
                                reward = 0
                            if frame == fr_end:
                                self.cont_actions_robot[action] += 1
                                self.CA_intime += 1
                                # print("Action idx: ", action_idx)
                                # print("*************** CORRECT ACTION (in time) ***************")
                                inaction.append("action")
                                self.update("action",action)
                        # Late
                        else:
                            if frame == fr_execution:
                                self.cont_actions_robot[action] += 1
                                self.CA_late += 1
                                self.reward_time += -1
                                # print("Action idx: ", action_idx)
                                # print("*************** CORRECT ACTION (late) ***************")
                                inaction.append("action")
                                frame_post.append(frame)
                                self.reward_energy = 0
                                reward = self.reward_time
                                self.update("action",action)

                            if frame >=  fr_end:
                                self.reward_time += -1

                    # # INCORRECT
                    else:
                        self.flags['freeze state'] = True
                        # INCORRECT ACTION
                        if self.flags["action robot"] == True:
                            if fr_execution <= fr_end:
                                if frame == fr_execution:
                                    frame_post.append(frame)

                                if frame == fr_end:
                                    self.IA_intime += 1
                                    # print("Action idx: ", action_idx)
                                    # print("*************** INCORRECT ACTION (in time) ***************")
                                    inaction.append("action")
                                    new_threshold, correct_action = self.select_correct_action(action)
                                    self.flags['evaluation'] = 'Incorrect action'
                                    self.energy_robot_reward(action)
                                    self.get_energy_robot_reward(action)
                                    self.reward_energy = self.reward_energy
                                    reward = self.reward_energy
                            else:
                                if frame > fr_end:
                                    self.reward_time += -1
                                if frame == fr_execution:
                                    self.IA_late += 1
                                    # print("Action idx: ", action_idx)
                                    # print("*************** INCORRECT ACTION (late) ***************")
                                    frame_post.append(frame)
                                    new_threshold, correct_action = self.select_correct_action(action)
                                    self.flags['evaluation'] = 'Incorrect action'
                                    self.get_energy_robot_reward(action)
                                    self.reward_energy = self.reward_energy
                                    reward = self.reward_energy + self.reward_time

                        # UNNECESARY ACTION
                        else:
                            if fr_execution <= fr_end:

                                if frame == fr_execution:
                                    frame_post.append(frame)

                                if frame == fr_end:

                                    inaction.append("action")
                                    self.energy_robot_reward(action)
                                    self.get_energy_robot_reward(action)
                                    reward = self.reward_energy
                                    if reward == 0: # action done very anticipated
                                        self.cont_actions_robot[action] += 1
                                        self.UAC_intime += 1
                                        self.update("action",action)
                                        # print("*************** LONG-TERM CORRECT ACTION ***************")
                                    else:
                                        self.UAI_intime += 1
                                        # print("*************** UNNECESARY ACTION ***************")
                                    #12345
                                        if recipe == 'c' or recipe == 'd':
                                            if action == 2: self.UA_related += 1
                                            else: self.UA_unrelated += 1
                                        elif recipe == 't':
                                            if action in [0, 1, 3, 4]: self.UA_related += 1
                                            else: self.UA_unrelated += 1
                                        self.update("unnecesary",action)

                            else:
                                if frame > fr_end:
                                    self.reward_time += -1
                                if frame == fr_execution:
                                    # print("Action idx: ", action_idx)
                                    
                                    inaction.append("action")
                                    frame_post.append(frame)
                                    self.energy_robot_reward(action)
                                    self.get_energy_robot_reward(action)
                                    reward =  self.reward_energy + self.reward_time
                                    if self.reward_energy == 0: # action done very anticipated
                                        self.cont_actions_robot[action] += 1
                                        self.UAC_late += 1
                                        self.update("action",action)
                                        # print("*************** LONG-TERM CORRECT ACTION (late) ***************")
                                    else:
                                        self.UAI_late += 1
                                        # print("*************** UNNECESARY ACTION (late) ***************")
                                        #12345
                                        if recipe == 'c' or recipe == 'd':
                                            if action == 2: self.UA_related += 1
                                            else: self.UA_unrelated += 1
                                        elif recipe == 't':
                                            if action in [0, 1, 3, 4]: self.UA_related += 1
                                            else: self.UA_unrelated += 1
                                        self.update("unnecesary",action)
                                    if  self.flags['threshold'] == 'next action init'   :
                                        self.flags['threshold'] = 'next action'
                                    
                                    self.flags['break'] = True

                else:
                    inaction.append([action, self.state, reward])
                    if self.flags['threshold'] == '':
                        optim = False
                    elif self.flags['decision'] == True:
                        if frame == fr_end:
                            # CORRECT INACTION
                            if simple_reward > -1:
                                self.cont_actions_robot[action] += 1
                                # no se actualiza como antes
                                self.CI += 1
                                # print("*************** CORRECT INACTION ***************")
                                reward = self.select_inaction_sample(inaction)

                            # INCORRECT INACTION
                            else:
                                self.flags['freeze state'] = True
                                self.II += 1
                                new_threshold, correct_action = self.select_correct_action(action)
                                reward = 0
                                self.flags['evaluation'] = 'Incorrect inaction'
                                # print("*************** INCORRECT INACTION ***************")
                            frame_post.append(frame)

                        elif fr_end < fr_execution: # special case two consecutive actions
                            if frame == fr_execution:
                                if simple_reward == -1:
                                    self.II += 1
                                    new_threshold, correct_action = self.select_correct_action(action)
                                    self.reward_time = -(fr_execution -fr_end)
                                    reward = self.reward_time
                                    self.flags['evaluation'] = 'Incorrect inaction'
                                    # pdb.set_trace()
                                    frame_post.append(frame)
                        else:
                            optim = False
                    else:
                        optim = False
              

        return reward, new_threshold, optim, frame_post, correct_action

    def prints_terminal(self, action, frame_prev, frame_post, reward):

        global annotations

        person_states_index = annotations['label']
        fr_init = annotations['frame_init']
        fr_end = annotations['frame_end']

        person_states = []

        for idx,val in enumerate(person_states_index):
            person_states.append(ATOMIC_ACTIONS_MEANINGS[val])

        data = {"States": person_states, "Frame init": fr_init, "Frame end": fr_end}

        df = pd.DataFrame(data)

        accion_robot = ROBOT_ACTIONS_MEANINGS[action]

        data_robot = {"Robot action":accion_robot, "Frame init": int(frame_prev), "Frame end": str(frame_post), "Reward": reward, "Reward time": self.reward_time, "Reward energy": self.reward_energy, "Time execution": self.time_execution}
        # pdb.set_trace()
        df_robot = pd.DataFrame(data_robot, index=[0])

        print("----------------------------------- Video -----------------------------------")
        print(df)
        print("\n----------------------------------- Robot -----------------------------------")
        print(df_robot)

    def prints_debug(self, action):
        global annotations, action_idx, memory_objects_in_table, frame

        person_states_index = annotations['label']
        fr_init = annotations['frame_init']
        fr_end = annotations['frame_end']

        person_states = []

        for idx,val in enumerate(person_states_index):
            person_states.append(ATOMIC_ACTIONS_MEANINGS[val])

        data = {"States": person_states, "Frame init": fr_init, "Frame end": fr_end}

        df = pd.DataFrame(data)
        state_prev = ATOMIC_ACTIONS_MEANINGS[annotations['label'][action_idx-1]]
        state = ATOMIC_ACTIONS_MEANINGS[annotations['label'][action_idx]]

        variations_in_table = len(memory_objects_in_table)
        if variations_in_table < 2:
            oit_prev = memory_objects_in_table[0]
            oit = memory_objects_in_table[0]
        else:
            oit_prev = memory_objects_in_table[variations_in_table-1]
            oit = memory_objects_in_table[variations_in_table]

        objects_prev_print = []
        for key,value in OBJECTS_MEANINGS.items():
            if oit_prev[key] == 1:
                objects_prev_print.append(value)

        objects_print = []
        for key,value in OBJECTS_MEANINGS.items():
            if oit[key] == 1:
                objects_print.append(value)

        action_robot = ROBOT_ACTIONS_MEANINGS.copy()[action]

        set1 = set(objects_prev_print)
        set2 = set(objects_print)

        missing = list(sorted(set1 - set2))
        added = list(sorted(set2 - set1))

        if len(missing) > 0:
            change = missing[0]
        elif len(added) > 0:
            change = added[0]
        else:
            change = 'nothing'

        print("Frame: ", str(frame))
        print("     ",state_prev)
        print("\nOBJECTS IN TABLE:")
        for obj in objects_prev_print:
            if obj == change and action !=5:
                print('----> ' + obj + ', ')
            else:
                print(obj + ', ')

        print("\n ROBOT ACTION: \n", action_robot)
        print("-------- State Video --------")
        print("     ",state)
        print("OBJECTS IN TABLE:")
        for obj in objects_print:
            if obj == change and action !=5:

                print('----> ' + obj + ', ')
            else:
                print(obj + ', ')


    def step(self, array):
        """
        Transition from the current state (self.state) to the next one given an action.

        Input:
            action: (int) action taken by the agent.
        Output:
            next_state: (numpy array) state transitioned to after taking action.
            reward: (int) reward received.
            done: (bool) True if the episode is finished (the recipe has reached its end).
            info:
        """
        global frame, action_idx, annotations, inaction, memory_objects_in_table, correct_action, path_labels_pkl,ROBOT_ACTION_DURATIONS, FACTOR_ENERGY_PENALTY,ERROR_PROB, healthy_human


        action_array = array[0]
        array_conf = array[1]
        self.flags['decision'] = action_array[1]
        action = action_array[0]
        self.mode = action_array[2]
        ROBOT_ACTION_DURATIONS = array_conf[1]
        FACTOR_ENERGY_PENALTY = array_conf[0]
        ERROR_PROB = array_conf[2]
        healthy_human = array_conf[3]
        # pdb.set_trace()
        assert self.action_space.contains(action)
        reward = 0
        self.reward_energy = 0
        self.reward_time = 0
        path_env = 0
        done = False
        optim = False

        self.flags['freeze state'] = False
        self.flags['pdb'] = False
        self.flags['break'] = False
        self.flags['evaluation'] = 'Not evaluated'
        self.flags['threshold'] = " "

        len_prev = 1
        min_time = 0 #Optimal time for recipe
        max_time = 0 #Human time withour HRI
        hri_time = 0

        threshold, fr_execution, fr_end = self.time_course(action)


        """
        if self.flags['decision']:
            print("\nFrame. ", frame)
            print("Threshold: ", threshold)
            print("Fr execution: ", fr_execution)
            print("Fr end: ", fr_end)
        """

        frame_prev = frame
        frame_post = []
        orig_action = action
        pre_action_idx = action_idx
        # print('action idx: ',action_idx)

        if not self.df_actions_to_be_done_robot_video.empty:
            for idx,row in self.df_actions_to_be_done_robot_video.iterrows():
                if row['action_idx']==0:
                    if self.objects_in_table[row['Object']] == 0: # lo hizo la persona
                        if frame > annotations['frame_end'][0]:
                            self.objects_in_table[row['Object']]=1
                if row['action_idx']>1:
                    if row['Frame end'] < frame:
                        if self.objects_in_table[row['Object']] == 0:
                            self.flags['freeze state'] = True
                            action_idx = row['action_idx']
                            if self.flags['decision'] == True:
                                # threshold, fr_execution, fr_end = self.time_course(action)
                                # fr_end = frame
                                if action == 5:
                                    self.flags['threshold'] = "second"
                                self.flags['threshold'] = "parada"
                                self.flags['freeze state'] = True
                                self.reward_time += -(frame-int(annotations['frame_init'][action_idx]))

        if action !=5:
            self.update_objects_in_table(action)
            memory_objects_in_table.append(list(self.objects_in_table.values()))
        if frame >= annotations['frame_init'].iloc[-1]:
            while frame <= annotations['frame_end'].iloc[-1]:

                if annotations['frame_init'][action_idx] <= frame <= annotations['frame_end'][action_idx]:
                    self.person_state = ATOMIC_ACTIONS_MEANINGS[annotations['label'][action_idx]]

                self.robot_state = "Predicting..."
                self.transition()
                self.rwd_history.append([reward]) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
                self.h_history.append([self.person_state])
                self.r_history.append([self.robot_state])
                self.rwd_time_h.append([self.reward_time])
                self.rwd_energy_h.append([self.reward_energy])
                next_state = self.state
            self.save_history()
            done = True

        else:
            while frame <= threshold:
                current_state = self.state #Current state
                if self.flags['decision'] == False:
                    # se transiciona de estado pero no se hace ninguna acción
                    # self.flags['freeze state'] = False #############
                    self.robot_state = "Predicting..."
                    if annotations['frame_init'][action_idx-1] <= frame <= annotations['frame_end'][action_idx-1]:
                        self.person_state = ATOMIC_ACTIONS_MEANINGS[annotations['label'][action_idx-1]]
                    else:
                        self.person_state = "other manipulation"
                    self.rwd_history.append([reward]) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
                    self.h_history.append([self.person_state])
                    self.r_history.append([self.robot_state])
                    self.rwd_time_h.append([self.reward_time])
                    self.rwd_energy_h.append([self.reward_energy])
                    self.transition() #Transition to a new state
                    next_state = self.state

                else:
                    optim = True
                    self.flags['freeze state']  = True
                    reward, new_threshold, optim, frame_post, correct_action = self.evaluation(action, fr_execution, fr_end, frame_post)
                    if new_threshold != 0:
                        threshold = new_threshold
                        fr_execution = new_threshold
                        action = correct_action
                        self.update_objects_in_table(action)
                        memory_objects_in_table.append(list(self.objects_in_table.values()))
                        len_prev = 2

                    if action != 5:
                        self.robot_state = ROBOT_ACTIONS_MEANINGS[action]
                    else:
                        self.robot_state = "Predicting..."

                    # if frame > fr_execution:
                    #     fr_execution = frame
                    # frame_end_annotations = annotations['frame_end'][action_idx-1]
                    if fr_execution <= fr_end:
                        if annotations['frame_init'][action_idx-1] <= frame <= annotations['frame_end'][action_idx-1]:
                            self.person_state = ATOMIC_ACTIONS_MEANINGS[annotations['label'][action_idx-1]]
                        else:
                            self.person_state = "other manipulation"
                        if frame > fr_execution:
                            self.robot_state = "Waiting for evaluation..."
                    elif fr_execution > fr_end:
                        # pdb.set_trace()
                        if frame > fr_end:
                            self.person_state = "Waiting for robot action..."
                        elif frame <= fr_end:
                            if annotations['frame_init'][action_idx-1] <= frame <= annotations['frame_end'][action_idx-1]:
                                self.person_state = ATOMIC_ACTIONS_MEANINGS[annotations['label'][action_idx-1]]
                            else:
                                self.person_state = "other manipulation"
                    # if self.person_state == "other manipulation" and self.robot_state == "Waiting for evaluation...":
                    #     print('action: ', action)
                        # pdb.set_trace()
                        
                    if frame == threshold :
                        self.flags['freeze state']  = False

                    self.rwd_history.append([reward]) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
                    self.h_history.append([self.person_state])
                    self.r_history.append([self.robot_state])
                    self.rwd_time_h.append([self.reward_time])
                    self.rwd_energy_h.append([self.reward_energy])
                    self.transition() #Transition to a new state
                    next_state = self.state


                #PRINT STATE-ACTION TRANSITION & REWARD
                if self.display: self.render(current_state, next_state, action, reward, self.total_reward)
                if frame >= annotations['frame_end'].iloc[-1]:
                    done = True
                if self.flags['break'] == True:
                    break

        if self.flags['decision'] == True:
            if orig_action != action:
                self.debug_actions_recipe.append(ROBOT_ACTIONS_MEANINGS[orig_action])
                self.debug_actions_recipe.append(ROBOT_ACTIONS_MEANINGS[action])
            else:
                self.debug_actions_recipe.append(ROBOT_ACTIONS_MEANINGS[action])
            self.state[110:133] = memory_objects_in_table[len(memory_objects_in_table)-1]

        # if optim == True:
            # if action != 5:
                # print('Frame prev: ',frame_prev)
                # # print('Pre action idx: ',pre_action_idx)
                # # print(self.flags)
                # self.prints_terminal(action, frame_prev, frame_post, reward)
                # print('Frame post: ',frame)
            #     print('action idx: ' ,action_idx)
            #     if self.reward_time != 0:
            #         pdb.set_trace()
                # print('Post action idx: ', action_idx)
        if done == True:
            self.debug_actions_recipe = []
            # pdb.set_trace()
            # path_to_save = videos_realData[video_idx] + '/human_times'
            # human_times = np.load(path_to_save, allow_pickle = True)
            # minimum_time_video = human_times['min']
            
            # if self.time_execution < minimum_time_video:
            #     self.prints_terminal(action, frame_prev, frame_post, reward)
            #     print('minimum time video: ',minimum_time_video)
            #     print('time execution: ', self.time_execution)
                
            #     pdb.set_trace()
            '''
            total_minimum_time_execution, _ =self.get_minimum_execution_times()
            print(annotations)
            print("Here, at the done: ", videos_realData[video_idx])
            # print("Execution timees: ", execution_times[0])
            path_to_save = videos_realData[video_idx] + '/human_times'
            print("Path: ", path_to_save)
            human_times = {'min': total_minimum_time_execution, 'human_time': int(annotations['frame_end'].iloc[-1])}
            with open(path_to_save, 'wb') as handle:
                pickle.dump(human_times, handle, protocol=pickle.HIGHEST_PROTOCOL)
            '''
            if self.idles_list:
               self.anticipation = np.mean(self.idles_list) * cfg.DECISION_RATE / 30 #In seconds
               self.anticipation_var = np.var(self.idles_list) * cfg.DECISION_RATE / 30 #In seconds
               self.acticipation_std = np.std(self.idles_list)* cfg.DECISION_RATE / 30 
            else:
               self.anticipation = 0.0
               self.anticipation_var = 0.0
               self.anticipation_std = 0.0
               
        self.total_reward += reward
       

        return self.state, reward, done, optim,  self.flags['pdb'], self.reward_time, self.reward_energy, self.time_execution, action, self.flags['threshold'], self.prediction_error, self.total_prediction


    def get_total_reward(self):
        return self.total_reward



    def save_history(self):
        """
        Saves the history of states/actions for the robot and the human in a .npz file.
        """

        if len(self.h_history) > 0 and self.test:
            path = './temporal_files/History_Arrays/'
            if not os.path.exists(path): os.makedirs(path)

            file_name = "{0}.npz".format(video_idx)
            np.savez(os.path.join(path, file_name),
            h_history=self.h_history,
            r_history=self.r_history,
            rwd_history=self.rwd_history,
            rwd_time_h = self.rwd_time_h,
            rwd_energy_h = self.rwd_energy_h
            )


    def reset(self):
        """
        Resets the environment to an initial state.
        """
        super().reset()

        global video_idx, action_idx, annotations,annotations_orig, frame, inaction, memory_objects_in_table, path_labels_pkl, recipe


        
        inaction = []
        memory_objects_in_table = []

        self.time_execution = 0
        self.reward_energy = 0
        self.reward_time = 0
        
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!

        if video_idx+1 < total_videos:
            video_idx += 1

        else:
            video_idx = 0
            #print("EPOCH COMPLETED.")

        #print("Video idx in reset: ", video_idx)

        #annotations = np.load(videos[video_idx], allow_pickle=True)
        action_idx = 1
        frame = 0

        # FOR REAL DATA --------------------------------------------------------------- !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # 0) Identify the recipe 123456
        recipe = videos_realData[video_idx].split("/")[-1][0]

        # 1) Read labels and store it in annotation_pkl
        labels_pkl = 'labels_updated.pkl'
        path_labels_pkl = os.path.join(videos_realData[video_idx], labels_pkl)
        
        annotations_orig = np.load(path_labels_pkl, allow_pickle=True)
        annotations = np.load(path_labels_pkl, allow_pickle=True)
        
        # 2) Read initial state
        frame_pkl = 'frame_0000' #Initial frame pickle
        path_frame_pkl = os.path.join(videos_realData[video_idx], frame_pkl)
        if NORMALIZE_DATA:
            read_state = np.load(path_frame_pkl, allow_pickle=True)
    
            ac_pred = (read_state['data'][0:33] - cfg.MEAN_ACTION_PREDICTION) / cfg.STD_ACTION_PREDICTION
            ac_rec = (read_state['data'][33:66] - cfg.MEAN_ACTION_RECOGNITION) / cfg.STD_ACTION_RECOGNITION
            vwm = (read_state['data'][66:110] - cfg.MEAN_VWM) / cfg.STD_VWM
            
            z = (read_state['z'] - cfg.MEAN_Z) / cfg.STD_Z
            
            oit = list(OBJECTS_INIT_STATE.values())
            oit = (oit - np.mean(oit)) / np.std(oit)
            # pdb.set_trace()

        else:
        #print("\n\nPath to annotation pkl: ", path_labels_pkl)


        #print("This is the annotation pkl: \n", annotations)
        #print("Which is of length: ", len(annotation_pkl[0]))

            read_state = np.load(path_frame_pkl, allow_pickle=True)
    
            ac_pred = read_state['data'][0:33]
            ac_rec = read_state['data'][33:66]
            vwm = read_state['data'][66:110]
            z = read_state['z']
            oit = list(OBJECTS_INIT_STATE.values())
            # pre_softmax = read_state['pre_softmax']
    
            # data[0:33] = pre_softmax

        if ONLY_RECOGNITION==True:
            if Z_hidden_state:
                self.state = np.concatenate((ac_rec,vwm,oit,z))
            else:
                self.state = np.concatenate((ac_rec,vwm,oit))
        else:
            if Z_hidden_state:
                self.state = np.concatenate((ac_pred,ac_rec,vwm,oit,z))
            else:
                self.state = np.concatenate((ac_pred,ac_rec,vwm,oit))
           

        self.total_reward = 0
        self.CA_intime = 0
        self.CA_late = 0
        self.IA_intime = 0
        self.IA_late = 0
        self.UAC_intime = 0
        self.UAC_late = 0
        self.UAI_intime = 0
        self.UAI_late = 0
        self.CI = 0
        self.II = 0
        self.cont_actions_robot = cfg.ROBOT_CONT_ACTIONS_MEANINGS.copy()

        #12345
        self.UA_related = 0
        self.UA_unrelated = 0

        self.prediction_error = 0
        self.total_prediction = 0


        self.r_history = []
        self.h_history = []
        self.rwd_history = []
        self.rwd_time_h = []
        self.rwd_energy_h = []
        self.idles_list = []
        self.debug_actions_recipe = []

        self.objects_in_table = OBJECTS_INIT_STATE.copy()
        memory_objects_in_table.append(list(self.objects_in_table.values()))

        ## check which actions has to do the robot at the starting of each video
        _,df_video = self.get_minimum_execution_times()

        #filtro para quedarme solo con las acciones que realmente se pueden hacer
        self.df_actions_to_be_done_robot_video = pd.DataFrame()
        if not df_video.empty:
            df_video_ = df_video[df_video['In table']==1]
            self.df_actions_to_be_done_robot_video = df_video_.copy()

        return self.state


    def _take_action(self, action, state = []):
        global annotations, action_idx
        """
        Version of the take action function that considers a unique correct robot action for each state, related to the required object and its position (fridge or table).

        Input:
            action: (int) from the action repertoire taken by the agent.
        Output:
            reward: (int) received from the environment.

        """

        global memory_objects_in_table
        if state == []:
            state = undo_one_hot(self.state[0:33]) #Next action prediction

        object_before_action = memory_objects_in_table[len(memory_objects_in_table)-2]
        reward = 0
        positive_reward = POSITIVE_REWARD

        self.total_prediction += 1

        if annotations['label'][action_idx] != state:
            self.prediction_error += 1
            state = annotations['label'][action_idx]
            # pdb.set_trace()

        self.flags["action robot"] = False

        if state == 1: #'pour milk'

            if action ==5: #'bring milk'
                reward = positive_reward

            else: reward = -1

        elif state == 2: #'pour water'

            if action ==5: #'bring water'
                reward = positive_reward
            else: reward = -1

        elif state == 3: #'pour coffee'
            if action ==5: #'do nothing' -> *coffee is at arm's reach
                reward = positive_reward
            else: reward = -1

        elif state == 4: #'pour Nesquik'
            if action ==5:
                reward = positive_reward
            else: reward = -1

        elif state == 5: #'pour sugar'
            if action ==5:
                reward = positive_reward
            else: reward = -1

        elif state ==6: #'put microwave'
            if action ==5:
                reward = positive_reward
            else: reward = -1

        elif state == 7: #'stir spoon
            if action ==5:
                reward = positive_reward
            else: reward = -1

        elif state == 8: #'extract milk fridge'

            
            key = [k for k, v in OBJECTS_MEANINGS.items() if v == 'milk'][0]
            
            if object_before_action[key] == 0:
                self.flags["action robot"] = True
                if action == 2:
                    # print("ENTRA")
                    reward = positive_reward
                else: 
                    reward = -1
            elif object_before_action[key] == 1:

                # print("AQUI TB")
                self.flags["action robot"] = False
                if action ==5:
                    # self.update("action")
                    # reward = 5 ############################################
                    reward = positive_reward
                else:
                    reward = -5
            

        elif state == 9: #'extract water fridge'
            # self.flags["action robot"] = True
            if action ==5:
                reward = positive_reward
            else: reward = -1

        elif state == 10: #'extract sliced bread'
            if action ==5:
                reward = positive_reward
            else: reward = -1

        elif state == 11: #'put toaster'
            if action ==5:
                reward = positive_reward
            else: reward = -1

        elif state == 12: #'extract butter fridge'
            # pdb.set_trace()
            self.flags["action robot"] = True
            key = [k for k, v in OBJECTS_MEANINGS.items() if v == 'butter'][0]
            if object_before_action[key] == 0:
                if action == 0: #'bring butter'
                    reward = positive_reward
                else: reward = -1
                #pdb.set_trace()
            elif object_before_action[key] == 1:
                # pdb.set_trace()
                self.flags["action robot"] = False
                if action ==5:
                    reward = positive_reward
                    # reward = 5
                else:
                    reward = -5
            

        elif state == 13: #'extract jam fridge'
            self.flags["action robot"] = True
            key = [k for k, v in OBJECTS_MEANINGS.items() if v == 'jam'][0]
            
            if object_before_action[key] == 0:
                if action == 1: #'bring jam'
                    reward = positive_reward
                else: reward = -1
            elif object_before_action[key] == 1:
                # pdb.set_trace()
                self.flags["action robot"] = False
                if action ==5:
                    reward = positive_reward
                    # reward = 5
                else:
                    reward = -5
            
            # print(reward)

        elif state == 14: #'extract tomato sauce fridge'
            self.flags["action robot"] = True
            key = [k for k, v in OBJECTS_MEANINGS.items() if v == 'tomato sauce'][0]
            
            if object_before_action[key] == 0:
                if action == 4: #'bring tomato sauce'
                    reward = positive_reward
                else: reward = -1
            elif object_before_action[key] == 1:
                # pdb.set_trace()
                self.flags["action robot"] = False
                if action == 5:
                    # reward = 5
                    reward = positive_reward
                else:
                    reward = -5
            

        elif state == 15: #'extract nutella fridge'
            self.flags["action robot"] = True
            key = [k for k, v in OBJECTS_MEANINGS.items() if v == 'nutella'][0]
            if object_before_action[key] == 0:
                if action == 3: #'bring nutella'
                    reward = positive_reward
                else: reward = -1
            elif object_before_action[key] == 1:
                # pdb.set_trace()
                self.flags["action robot"] = False
                if action == 5:
                    reward = positive_reward
                    # reward = 5
                else:
                    reward = -5
            

        elif state == 16: #'spread butter'
            if action == 5:
                reward = positive_reward
            else: reward = -1

        elif state == 17: #'spread jam'
            if action == 5:
                reward = positive_reward
            else: reward = -1

        elif state == 18: #'spread tomato sauce'
            if action ==5:
                reward = positive_reward
            else: reward = -1

        elif state == 19: #'spread nutella'
            if action ==5:
                reward = positive_reward
            else: reward = -1

        elif state == 20: #'pour olive oil'
            if action ==5:
                reward = positive_reward
            else: reward = -1

        elif state == 21: #'put jam fridge'
            #self.flags["action robot"] = True
            #key = [k for k, v in OBJECTS_MEANINGS.items() if v == 'jam'][0]
            if action ==5:
                reward = positive_reward

            else: reward = -1
            """
            elif object_before_action[key] == 0:
                # pdb.set_trace()
                self.flags["action robot"] = False
                if action ==5:
                    reward = 5
                else:
                    reward = -5
            """
            #else: reward = -1

        elif state == 22: #'put butter fridge'
            #self.flags["action robot"] = True
            #key = [k for k, v in OBJECTS_MEANINGS.items() if v == 'butter'][0]
            if action ==5:
                reward = positive_reward

            else: reward = -1
            """
            elif object_before_action[key] == 0:
                # pdb.set_trace()
                self.flags["action robot"] = False
                if action ==5:
                    reward = 5
                else:
                    reward = -5
            """
            #else: reward = -1

        elif state == 23: #'put tomato sauce fridge'
            #self.flags["action robot"] = True
            #key = [k for k, v in OBJECTS_MEANINGS.items() if v == 'tomato sauce'][0]
            if action ==5:
                reward = positive_reward

            else: reward = -1
            """
            elif object_before_action[key] == 0:
                # pdb.set_trace()
                self.flags["action robot"] = False
                if action ==5:
                    reward = 5
                else:
                    reward = -5
            """
            #else: reward = -1

        elif state == 24: #'put nutella fridge'
            #self.flags["action robot"] = True
            #key = [k for k, v in OBJECTS_MEANINGS.items() if v == 'nutella'][0]
            if action ==5:
                reward = positive_reward

            else: reward = -1

            """
            elif object_before_action[key] == 0:
                # pdb.set_trace()
                self.flags["action robot"] = False
                if action ==5:
                    reward = 5
                else:
                    reward = -5
            """
            #else: reward = -1

        elif state == 25: #'pour milk bowl'
            if action ==5:
                reward = positive_reward
            else: reward = -1

        elif state == 26: #'pour cereals bowl'
            if action ==5:
                reward = positive_reward
            else: reward = -1

        elif state == 27: #'pour nesquik bowl'
            if action ==5:
                reward = positive_reward
            else: reward = -1

        elif state == 28: #'put bowl microwave'
            if action ==5:
                reward = positive_reward
            else: reward = -1

        elif state == 29: #'stir spoon bowl'
            if action ==5:
                reward = positive_reward
            else: reward = -1

        elif state == 30: #'put milk fridge'
            #self.flags["action robot"] = True
            #key = [k for k, v in OBJECTS_MEANINGS.items() if v == 'milk'][0]
            if action ==5:
                reward = positive_reward

            else: reward = -1
            """
            elif object_before_action[key] == 0:
                self.flags["action robot"] = False
                if action ==5:
                    reward = 5
                else:
                    reward = -5
            """
            #else: reward = -1

        elif state == 31: #'put sliced bread plate'
            if action ==5:
                reward = positive_reward
            else: reward = -1

        else:
            if action ==5: #'do nothing'
                reward = positive_reward
            else: reward = -1

        return reward


    def transition(self):
        """
        Gets a new observation of the environment based on the current frame and updates the state.

        Global variables:
            frame: current time step.
            action_idx: index of the NEXT ACTION (state as the predicted action). *The action_idx points towards the next atomic action at the current frame.
            annotations: pickle with the annotations, in the form of a table.

        """

        global action_idx, frame, annotations, inaction, memory_objects_in_table, video_idx


        frame += 1 #Update frame

        if frame <= annotations['frame_end'].iloc[-1]:
            self.time_execution += 1

        length = len(annotations['label']) - 1

        # if frame % cfg.DECISION_RATE != 0: return ### que coño es esto????????????????????????????????

        # 1)
        #GET TIME STEP () (Updates the action_idx)
        #We transition to a new action index when we surpass the init frame of an action (so that we point towards the next one).
        # print('freeze: ',self.flags['freeze state'])
        if  self.flags['freeze state'] == False:
            # frame >= annotations['frame_init'][action_idx]
            if  frame >= annotations['frame_init'][action_idx]:
                if action_idx <= length-1:
                    action_idx += 1
                    inaction = []


        # 2) GET STATE FROM OBSERVED DATA
        # 2.1 ) Read the pickle
        frame_to_read = int(np.floor(frame/6))
        frame_pkl = 'frame_' + str(frame_to_read).zfill(4) # Name of the pickle (without the .pkl extension)
        path_frame_pkl = os.path.join(videos_realData[video_idx], frame_pkl) # Path to pickle as ./root_realData/videos_realData[video_idx]/frame_pkl

        #print("This is the path to the frame picke: ", path_frame_pkl)

        read_state = np.load(path_frame_pkl, allow_pickle=True) # Contents of the pickle. In this case, a 110 dim vector with the Pred Ac., Reg. Ac and VWM
        #print("Contenido del pickle en el frame", frame, "\n", read_state)


        # 2.2 ) Generate state
        # data = read_state['data']
        # pre_softmax = read_state['pre_softmax']
        # data[0:33] = pre_softmax
        data = read_state['data'][:110] #[Ac pred + Ac reg + VWM]
        # OBJECTS IN TABLE
        variations_in_table = len(memory_objects_in_table)
        if variations_in_table < 2:

            oit = memory_objects_in_table[0]
        else:
            oit = memory_objects_in_table[variations_in_table-1]
            
        if NORMALIZE_DATA:
        
            ac_pred = (read_state['data'][0:33] - cfg.MEAN_ACTION_PREDICTION) / cfg.STD_ACTION_PREDICTION
            ac_rec = (read_state['data'][33:66] - cfg.MEAN_ACTION_RECOGNITION) / cfg.STD_ACTION_RECOGNITION
            vwm = (read_state['data'][66:110] - cfg.MEAN_VWM) / cfg.STD_VWM
            oit = (oit - np.mean(oit)) / np.std(oit)
            z = (read_state['z'] - cfg.MEAN_Z) / cfg.STD_Z
        else:
            ac_pred = read_state['data'][0:33]
            ac_rec = read_state['data'][33:66]
            vwm = read_state['data'][66:110] 
            
            z = read_state['z']



        if ONLY_RECOGNITION==True:
            if Z_hidden_state:
                self.state = np.concatenate((ac_rec,vwm,oit,z))
            else:
                self.state = np.concatenate((ac_rec,vwm,oit))
        else:
            if Z_hidden_state:
                self.state = np.concatenate((ac_pred,ac_rec,vwm,oit,z))
            else:
                self.state = np.concatenate((ac_pred,ac_rec,vwm,oit))




    def CreationDataset(self):
        #ver porque los states aveces tienen un elemento, de resto creo que esta todo ok
        global frame, action_idx, annotations_orig, ROBOT_ACTION_DURATIONS

        guarda = 10
        done = False
        state = []
        action = []
        no_action_state = []
        no_actions = []

        self.transition()
        fr_end = int(annotations['frame_end'][action_idx-1])

        # print("          Frame: ", frame)
        # print("Action idx: ",action_idx)
        # print(annotations['frame_end'].iloc[-2])

        if frame >= annotations['frame_end'].iloc[-2]:
            frame = annotations['frame_end'].iloc[-1]
            done = True


        df_video = self.get_possibility_objects_in_table(annotations_orig)
        # print(df_video)
        # pdb.set_trace()
        name_actions = []
        if df_video.empty:
            done = True
            state_env = state
            action_env  = 5
        else:
            for index, row in df_video.iterrows():
                if row['In table'] == 1:
                    name_actions.append("bring "+row['Object'])
                # else:
                #     name_actions.append("put "+row['Object']+ ' fridge')
    
            df_video_filtered = df_video[df_video['In table']==1]
            df_video_filtered['Actions'] = name_actions
    
            keys = list(df_video_filtered['Actions'])
            video_dict = {}
            for i in keys:
                video_dict[i] = 0
    
            person_states = annotations['label']
            df_video_dataset = df_video_filtered.copy()
    
            if not df_video_dataset.empty:

                for idx,value in enumerate(person_states):
                    for obj in INTERACTIVE_OBJECTS_ROBOT:
                        if obj in ATOMIC_ACTIONS_MEANINGS[value]:
    
                            if 'extract' in ATOMIC_ACTIONS_MEANINGS[value]:
                                if idx != 0:
                                    action_name = 'bring '+ obj
                                    fr_init = annotations['frame_init'][idx-1]
                                    df_video_dataset['Frame init'].loc[df_video_dataset['Actions']==action_name] = fr_init

                df_video_dataset.sort_values("Frame init")

            while frame < fr_end:
                if not df_video_dataset.empty:
                    for index, row in df_video_dataset.iterrows():
                            correct_action = [k for k, v in ROBOT_ACTIONS_MEANINGS.items() if v == row['Actions']]
                            correct_action = correct_action[0]
                            duration = cfg.ROBOT_ACTION_DURATIONS[correct_action]
                            if row['Frame init'] + guarda < frame < row['Frame end'] - duration:
                                while frame <  row['Frame end'] - duration:
                                    if video_dict[ROBOT_ACTIONS_MEANINGS[correct_action]] == 0:
                                        if 'tomato' in ROBOT_ACTIONS_MEANINGS[correct_action]:
                                            current_obj = 'tomato sauce'
                                        else:
                                            current_obj = ROBOT_ACTIONS_MEANINGS[correct_action].split(" ")[1]
                                        if 'bring' in ROBOT_ACTIONS_MEANINGS[correct_action]:
                                            if self.objects_in_table[current_obj] == 1:
                                                self.objects_in_table[current_obj] = 0
                                        memory_objects_in_table = list(self.objects_in_table.values())
                                        self.update_objects_in_table(correct_action)
                                        video_dict[ROBOT_ACTIONS_MEANINGS[correct_action]] = 1
    
                                    if len(memory_objects_in_table) > 1:
                                        state_append = self.state
                                        state_append[110:133] = memory_objects_in_table
                                        state.append(state_append)

                                    action.append(correct_action)
                                    self.transition()
                            else:
                                no_actions.append(5)
                                no_action_state.append(self.state)
                                self.transition()
                else:
                    no_actions.append(5)
                    no_action_state.append(self.state)
                    self.transition()

            if len(action)>0:
                new_no_actions = []
                new_no_actions_state = []

            else:
                number_of_no_actions = round(len(no_actions)*0.05)
                random_positions = random.sample(range(0,len(no_actions)),number_of_no_actions)
                new_no_actions = ([no_actions[i] for i in random_positions])
                new_no_actions_state = ([no_action_state[i] for i in random_positions])

            if len(action)>0:
                state_env = state
                action_env = action

            else:
                state_env= new_no_actions_state
                action_env= new_no_actions


        return state_env, action_env, done

    def get_video_idx(self):
        return video_idx

    def summary_of_actions(self):
        print("\nCORRECT ACTIONS (in time): ", self.CA_intime)
        print("CORRECT ACTIONS (late): ", self.CA_late)
        print("INCORRECT ACTIONS (in time): ", self.IA_intime)
        print("INCORRECT ACTIONS (late): ", self.IA_late)
        # print("UNNECESSARY ACTIONS (in time): ", self.UA_intime)
        # print("UNNECESSARY ACTIONS (late): ", self.UA_late)
        print("UNNECESSARY ACTIONS CORRECT (in time): ", self.UAC_intime)
        print("UNNECESSARY ACTIONS CORRECT (late): ", self.UAC_late)
        print("UNNECESSARY ACTIONS INCORRECT (in time): ", self.UAI_intime)
        print("UNNECESSARY ACTIONS INCORRECT (late): ", self.UAI_late)
        print("CORRECT INACTIONS: ", self.CI)
        print("INCORRECT INACTIONS: ", self.II)
        print("")

    def close(self):
        pass

