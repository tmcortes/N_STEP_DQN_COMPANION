#DQN PARAMETERS ------
#---------------------------------------------------------------------------------


REPLAY_MEMORY = 1000
 #Size of replay memory (deque object)

N_STEP = 3
NUM_EPOCH = 200
NUM_EPISODES = 63 #"Number of training epochs"
# BATCH_SIZE = int(round(REPLAY_MEMORY/20,0)) #128
BATCH_SIZE = 128
GAMMA = 0.99 #Discount rate for future rewards
EPS_START = 0.99 #Initial exporation rate
EPS_END = 0.1 #Final exploration rate
EPS_DECAY = NUM_EPOCH #Exploration rate decay factor
TARGET_UPDATE = 100 #Episodes between target network update (policy net parameters -> target net)
LR = 1e-4#Learning rate
POSITIVE_REWARD = 0
NO_ACTION_PROBABILITY = 80
FACTOR_ENERGY_PENALTY = 1
SPEED_ROBOT = 1
ERROR_PROB = 0
PROB_SAVE = 0
IMPORTANCE_SAMPLING = False
TAU = 0.005

ROOT = './Checkpoints/'
EXPERIMENT_NAME = "DQN"
SAVE_MODEL = False
SAVE_EPISODE = 100
LOAD_MODEL = False
LOAD_EPISODE = 0
REACTIVE = False

DECISION_RATE = 30 
Z_hidden_state = False
NORMALIZE_DATA = True
ONLY_RECOGNITION = False

PERSONALIZATION = False
PERSON = 'person1'
TYPE_PERSONALIZATION = 'full'
TEMPORAL_CONTEXT =False
LOOP = False
NUM_LOOP = 1
TEST_FOLD = 'fold1'
hidden_size_LSTM = 512
#ENVIRONMENT PARAMETERS ------
#---------------------------------------------------------------------------------

VERSION = 4 


INTERACTIVE_OBJECTS_ROBOT = ['butter','jam','milk','nutella','tomato sauce']

ATOMIC_ACTIONS_MEANINGS = {
	0: 'other manipulation',
	1: 'pour milk',
	2: 'pour water',
	3: 'pour coffee',
	4: 'pour Nesquik',
	5: 'pour sugar',
	6: 'put microwave',
	7: 'stir spoon',
	8: 'extract milk fridge',
	9: 'extract water fridge',
	10: 'extract sliced bread',
	11: 'put toaster',
	12: 'extract butter fridge',
	13: 'extract jam fridge',
	14: 'extract tomato sauce fridge',
	15: 'extract nutella fridge',
	16: 'spread butter',
	17: 'spread jam',
	18: 'spread tomato sauce',
	19: 'spread nutella',
	20: 'pour olive oil',
	21: 'put jam fridge',
	22: 'put butter fridge',
	23: 'put tomato sauce fridge',
	24: 'put nutella fridge',
	25: 'pour milk bowl',
	26: 'pour cereals bowl',
	27: 'pour nesquik bowl',
	28: 'put bowl microwave',
	29: 'stir spoon bowl',
	30: 'put milk fridge',
	31: 'put sliced bread plate',
	32: 'TERMINAL STATE',

}

"""
Objects that are in the fridge:
	butter
	jam
	milk
	nutella
	tomato sauce
	water
Objects that are at human's reach:
	everything else
"""

OBJECTS_MEANINGS = {
	0: 'background',
	1: 'bowl',
	2: 'butter',
	3: 'cereals',
	4: 'coffee',
	5: 'cup',
	6: 'cutting board',
	7: 'fork',
	8: 'fridge',
	9: 'jam',
	10: 'knife',
	11: 'microwave',
	12: 'milk',
	13: 'nesquik',
	14: 'nutella',
	15: 'olive oil',
	16: 'plate',
	17: 'sliced bread',
	18: 'spoon',
	19: 'sugar',
	20: 'toaster',
	21: 'tomato sauce',
	22: 'water'
}


OBJECTS_INIT_STATE = {
    'background': 0,
	'bowl': 1,
	'butter': 0,
	'cereals': 1,
	'coffee': 1,
	'cup': 1,
	'cutting board': 1,
	'fork': 1,
	'fridge': 1,
	'jam': 0,
	'knife': 1,
	'microwave': 1,
	'milk': 0,
	'nesquik': 1,
	'nutella': 0,
	'olive oil': 1,
	'plate': 1,
	'sliced bread': 1,
	'spoon': 1,
	'sugar': 1,
	'toaster': 1,
	'tomato sauce': 0,
	'water': 1
    } 

"""
ROBOT_ACTIONS_MEANINGS = {	
	0: 'bring bowl',
	1: 'bring butter',
	2: 'bring cereals',
	3: 'bring coffee',
	4: 'bring cup',
	5: 'bring fork',
	6: 'bring jam',
	7: 'bring knife',
	8: 'bring milk',
	9: 'bring nesquik',
	10: 'bring nutella',
	11: 'bring olive oil',
	12: 'bring plate',
	13: 'bring sliced bread',
	14: 'bring spoon',
	15: 'bring sugar',
	16: 'bring tomato sauce',
	17: 'bring water',
	18: 'do nothing',
	19: 'put jam fridge',
	20: 'put butter fridge',
	21: 'put tomato sauce fridge',
	22: 'put nutella fridge',
	23: 'put milk fridge'
}
"""

#Reduced action repertoire
# ROBOT_ACTIONS_MEANINGS = {
# 	0: 'bring butter',
# 	1: 'bring jam',
# 	2: 'bring milk',
# 	3: 'bring nutella',
# 	4: 'bring sliced bread',
# 	5: 'bring tomato sauce',
# 	6: 'do nothing',
# 	7: 'put jam fridge',
# 	8: 'put butter fridge',
# 	9: 'put tomato sauce fridge',
# 	10: 'put nutella fridge',
# 	11: 'put milk fridge'

# }

ROBOT_ACTIONS_MEANINGS = {
	0: 'bring butter',
	1: 'bring jam',
	2: 'bring milk',
	3: 'bring nutella',
	4: 'bring tomato sauce',
	5: 'do nothing',

}

ROBOT_CONT_ACTIONS_MEANINGS = {
	0: 0,
	1: 0,
	2: 0,
	3: 0,
	4: 0,
	5: 0,
}




# VERSION 1) AVERAGE OF HUMAN ACTION DURATIONS
# ROBOT_ACTION_DURATIONS = {
#  	0: 174,  # bring butter
#  	1: 198,  # bring jam
#  	2: 186,  # bring milk
#  	3: 234,  # bring nutella
#  	4: 270,  # bring tomato sauce
#  	5: 0,    # do nothing

#  }


ROBOT_AVERAGE_DURATIONS = {
 	0: 174,  # bring butter
 	1: 198,  # bring jam
 	2: 186,  # bring milk
 	3: 234,  # bring nutella
 	4: 270,  # bring tomato sauce
 	5: 0,    # do nothing

 }

"""  
# VERSION 2) 0.5*HUMAN ---> FAST ROBOT
ROBOT_ACTION_DURATIONS = {
	0: 87,  # bring butter
	1: 99,  # bring jam
	2: 93,  # bring milk
	3: 117,  # bring nutella
	4: 135,  # bring tomato sauce
	5: 0    # do nothing

}
# VERSION 3) 2*HUMAN ---> SLOW (MORE REALISTIC) ROBOT
ROBOT_ACTION_DURATIONS = {
	0: 348,  # bring butter
	1: 396,  # bring jam
	2: 372,  # bring milk
	3: 468,  # bring nutella
	4: 540,  # bring tomato sauce
	5: 0   # do nothing
}
"""

ROBOT_POSSIBLE_INIT_ACTIONS = {
	0: 1,
	1: 1,
	2: 1,
	3: 1,
	4: 1,
	5: 1,
}

#from aux import *
#ROBOT_ACTION_DURATIONS = get_estimations_action_time_human()


# ROBOT_ACTION_DURATIONS = ROBOT_AVERAGE_DURATIONS
# ROBOT_ACTION_DURATIONS.update((x, y*SPEED_ROBOT) for x, y in ROBOT_ACTION_DURATIONS.items())


N_OBJECTS = len(OBJECTS_MEANINGS)
ACTION_SPACE = len(ROBOT_ACTIONS_MEANINGS)
N_ATOMIC_ACTIONS = len(ATOMIC_ACTIONS_MEANINGS)

if Z_hidden_state:
    Z_HIDDEN = 1024
else:
    Z_HIDDEN = 0
    

#MEANS and STDs
MEAN_ACTION_PREDICTION = 0.030303033
STD_ACTION_PREDICTION = 0.15909983


MEAN_ACTION_RECOGNITION = 0.030303031
STD_ACTION_RECOGNITION = 0.16159552


MEAN_Z = -0.0092057185
STD_Z = 1.4593501


MEAN_VWM = [0.19343422, -0.13023123,   0.19343422, -0.13023123,   0.19343422, -0.13023123, 
  0.19343422, -0.13023123,   0.19343422, -0.13023123,   0.19343422, -0.13023123,
  0.19343422, -0.13023123,   0.19343422, -0.13023123,   0.19343422, -0.13023123,
  0.19343422, -0.13023123,   0.19343422, -0.13023123,   0.19343422, -0.13023123,
  0.19343422, -0.13023123,   0.19343422, -0.13023123,   0.19343422, -0.13023123,
  0.19343422, -0.13023123,   0.19343422, -0.13023123,   0.19343422, -0.13023123,
  0.19343422, -0.13023123,   0.19343422, -0.13023123,   0.19343422, -0.13023123,
  0.19343422, -0.13023123, ]

STD_VWM = [0.6616038,   0.69604784, 0.6616038,   0.69604784, 0.6616038,   0.69604784,
 0.6616038,   0.69604784, 0.6616038,   0.69604784, 0.6616038,   0.69604784,
 0.6616038,   0.69604784, 0.6616038,   0.69604784, 0.6616038,   0.69604784,
 0.6616038,   0.69604784, 0.6616038,   0.69604784, 0.6616038,   0.69604784,
 0.6616038,   0.69604784, 0.6616038,   0.69604784, 0.6616038,   0.69604784,
 0.6616038,   0.69604784, 0.6616038,   0.69604784, 0.6616038,   0.69604784,
 0.6616038,   0.69604784, 0.6616038,   0.69604784, 0.6616038,   0.69604784,
 0.6616038,   0.69604784,]


MEAN_VWM_EVEN = -0.13023123
STD_VWM_EVEN = 0.69604784

MEAN_VWM_ODD = 0.19343422
STD_VWM_ODD = 0.6616038

MEAN_TEMP_CTX = 177.0
STD_TEMP_CTX = 85.39906322671227

def print_setup(args):
	"""
	Prints a table with the arguments of the training script.
	Input:
		args: arguments during execution time. 
	
	"""
	print("")
	print(" Experiment ", args.experiment_name)
	print("="*39)
	print("  DQN parameters")
	print("="*39)
	print("| SIZE OF REPLAY MEMORY     | {0:<6g}".format(args.replay_memory), " |")
	print("| INITIAL EXPLORATION RATE  | {0:<6g}".format(args.eps_start), " |")
	print("| TERMINAL EXPLORATION RATE | {0:<6g}".format(args.eps_end), " |")
	print("| GAMMA DISCOUNT FACTOR     | {0:<6g}".format(args.gamma), " |")
	print("| FREQ. TARGET UPDATE       | {0:<6g}".format(args.target_update), " |")
	print("="*39)
	print("  Training parameters")
	print("="*39)
	print("| BATCH SIZE                | {0:<6g}".format(args.batch_size), " |")
	print("| LEARNING RATE             | {0:<6g}".format(args.lr), " |")
	print("| LOAD MODEL                | {0:<6g}".format(args.load_model), " |")
	print("| LOAD EPISODE              | {0:<6g}".format(args.load_episode), " |")
	
	print("="*39)	

