import sys, os
sys.path.insert(0, 'evoman')
from evoman.environment import Environment
from demo_controller import player_controller
from deap import tools, creator, base, algorithms
import numpy as np
import json
import multiprocessing


headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'test'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

player_life = 100
enemy_life = 100
dom_u = 1
dom_l = -1
n_population = 50 #100
gens = 50
mate = 1 #cxpb
mutation = 0.4 #matpb
last_best = 0
n_hidden_neurons = 15 #number of possible actions
budget = 500
enemies = 1
runs = 10
envs = []
eatype = "Roulette"
# initializes environment with ai player using random controller, playing against static enemy