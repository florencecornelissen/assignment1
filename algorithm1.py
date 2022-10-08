################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os
sys.path.insert(0, 'evoman') 
from environment import Environment
import numpy as np
import random
import neat
from controller import Controller
import pickle
import visualize


headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


class player_controller(Controller):
        
    def __init__(self, config):
        self.config = config

    def control(self, inputs, controller):
        NN = neat.nn.FeedForwardNetwork.create(controller, self.config)
        # inputs = (inputs-min(inputs)) / float((max(inputs)-min(inputs)))
        output = NN.activate(inputs)

		# takes decisions about sprite actions
        if output[0] > 0.5:
            left = 1
        else:
            left = 0

        if output[1] > 0.5:
            right = 1
        else:
            right = 0

        if output[2] > 0.5:
            jump = 1
        else:
            jump = 0

        if output[3] > 0.5:
            shoot = 1
        else:
            shoot = 0

        if output[4] > 0.5:
            release = 1
        else:
            release = 0

        return [left, right, jump, shoot, release]


# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, '../../../config.txt')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)

experiment_name = 'dummy_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name,
                multiplemode='no',
                player_controller=player_controller(config),
                enemies=[3])


# runs simulation
def simulation(env, x):
    f,p,e,t = env.play(pcont=x)
    return f

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = simulation(env, genome)


def run():

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    winner = pop.run(eval_genomes, 5)

    with open('../../../winner', 'wb') as f:
        pickle.dump(winner, f)

    visualize.plot_stats(stats, ylog=True, view=True, filename="../../../feedforward-fitness.svg")
    visualize.plot_species(stats, view=True, filename="../../../feedforward-speciation.svg")

    node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}
    visualize.draw_net(config, winner, True, node_names=node_names)

    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward.gv")
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward-enabled-pruned.gv", prune_unused=True)



run()