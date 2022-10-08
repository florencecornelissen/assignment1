import sys, os
sys.path.insert(0, 'evoman') 
from evoman.environment import Environment
from demo_controller import player_controller
from deap import tools, creator, base, algorithms
import numpy as np
from itertools import repeat
import math
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
gens = 5
mate = 1
mutation = 0.5
last_best = 0
n_hidden_neurons = 15 #number of possible actions
budget = 500
enemies = 1
runs = 10
envs = []
eatype = "Roulette"
# initializes environment with ai player using random controller, playing against static enemy

# (self-)adaptive mutation
def normal_dist(x, mean, std):
    #mean = np.mean(x)
    #std = np.std(x)
    ans = (np.pi * std) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    ans1 = 1/(std * math.sqrt(2*std)) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    return ans

def self_adaptive(individual,sigma,gens):
    sigma = 1 - 0.9 * g/gens
    size = len(individual)
    sigmanew = sigma * np.exp(normal_dist(gene, 0, sigma))
    genenew = gene + normal_dist(gene, 0, sigmanew)
    return individual



for enemy in range(1, 9):

    env = Environment(experiment_name=experiment_name,
                            enemies=[enemy],
                            player_controller=player_controller(n_hidden_neurons),
                            speed="fastest",
                            logs='off'
                            )

    n_weights = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

    creator.create('FitnessBest', base.Fitness, weights = (1.0,))
    creator.create('Individual', np.ndarray, fitness = creator.FitnessBest, player_life = player_life, enemy_life = enemy_life)

    tb = base.Toolbox() # contains the evolutionary operators

    
    tb.register('indices', np.random.uniform, dom_l,dom_u) 
    # initRepeat: Call the function container (creator.Individual) with a generator function (tb.indiceis) corresponding to the calling n (weights) times the function func.
    tb.register('individual', tools.initRepeat, creator.Individual, tb.indices, n = n_weights)
    tb.register('population', tools.initRepeat, list, tb.individual, n = n_population)


    def simulation(env,x):
        fitness,p_l,e_l,time_ = env.play(pcont=x)
        return fitness, e_l

    def evaluate(x,env):
        tmp, e_l = simulation(env, x)
        return tmp, e_l


    def evalpop(pop,env):
        to_evaluate_ind = [ind for ind in pop if not ind.fitness.valid]
        tmp = [env for i in range(len(to_evaluate_ind))]
        fits = tb.map(tb.evaluate,to_evaluate_ind,tmp)
        fitnesses, enemy_lifes = [], []
        for ind, (fit, e_l) in zip(pop, fits):
            ind.fitness.values = (fit,)
            enemy_lifes.append(e_l)
            fitnesses.append(fit)
        return (pop, fitnesses, enemy_lifes)           
            
        
    tb.register("evaluate", evaluate)
    tb.register("mate", tools.cxTwoPoint) # crossover operator
    tb.register("mutate", tools.mutShuffleIndexes, indpb=0.5)
    tb.register("select", tools.selBest)

    record = {}
    log = tools.Logbook()
    log.header = ['enemy','run','gen','individuals','mean','max']


    pop = tb.population(n=n_population)
    pop, fitnesses, enemy_lifes = evalpop(pop,env)
    hof = tools.HallOfFame(1, similar=np.array_equal)

    print(f'Enemy {enemy}')
    for g in range(1,gens):
        ii = np.argmax(fitnesses)
        print(f'{g} - max: {fitnesses[ii]} (health: {enemy_lifes[ii]}) - mean: {np.mean(fitnesses)}')
        pop = tb.select(pop,len(pop))
        offs = algorithms.varAnd(pop,tb,mate,mutation)
        offs, fitnesses, enemy_lifes = evalpop(offs,env)
        pop = offs