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
n_population = 100  # 100
gens = 50
mate = 0.6 #reproduction rate is 1 - mate - mutation
mutation = 0.4 #mate+mutation need to be <= 1.0
last_best = 0
n_hidden_neurons = 20  # number of possible actions
children = 100
mu = 10 #number individuals to select for next generation
budget = 500
enemies = 1
runs = 10
envs = []
eatype = "Roulette"
# initializes environment with ai player using random controller, playing against static enemy

for enemy in [4]:

    env = Environment(experiment_name=experiment_name,
                      enemies=[enemy],
                      player_controller=player_controller(n_hidden_neurons),
                      speed="fastest",
                      level=2,
                      logs='off'
                      )

    n_weights = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

    creator.create('FitnessBest', base.Fitness, weights=(1.0,))
    creator.create('Individual', np.ndarray, fitness=creator.FitnessBest, player_life=player_life,
                   enemy_life=enemy_life)

    tb = base.Toolbox()  # contains the evolutionary operators

    tb.register('indices', np.random.uniform, dom_l, dom_u)
    # initRepeat: Call the function container (creator.Individual) with a generator function (tb.indiceis) corresponding to the calling n (weights) times the function func.
    tb.register('individual', tools.initRepeat, creator.Individual, tb.indices, n=n_weights)
    tb.register('population', tools.initRepeat, list, tb.individual, n=n_population)


    def simulation(env, x):
        fitness, p_l, e_l, time_ = env.play(pcont=x)
        return fitness, e_l


    def evaluate(x, env):
        tmp, e_l = simulation(env, x)
        return tmp, e_l


    def evalpop(pop, env):
        to_evaluate_ind = [ind for ind in pop if not ind.fitness.valid]
        tmp = [env for i in range(len(to_evaluate_ind))]
        fits = tb.map(tb.evaluate, to_evaluate_ind, tmp)
        fitnesses, enemy_lifes = [], []
        for ind, (fit, e_l) in zip(pop, fits):
            ind.fitness.values = (fit,)
            enemy_lifes.append(e_l)
            fitnesses.append(fit)
        return (pop, fitnesses, enemy_lifes)


    tb.register("evaluate", evaluate)
    tb.register("mate", tools.cxTwoPoint)  # crossover operator
    tb.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.5)
    tb.register("select", tools.selTournament
                , tournsize=3)

    record = {}
    log = tools.Logbook()
    log.header = ['enemy', 'run', 'gen', 'individuals', 'mean', 'max']

    pop = tb.population(n=n_population)
    pop, fitnesses, enemy_lifes = evalpop(pop, env)
    hof = tools.HallOfFame(1, similar=np.array_equal)

    print(f'Enemy {enemy}')
    for g in range(1, gens):
        ii = np.argmax(fitnesses)
        print(f'{g} - max: {fitnesses[ii]} (health: {enemy_lifes[ii]}) - mean: {np.mean(fitnesses)}')
        pop = tb.select(pop, len(pop))
        offs = algorithms.varOr(pop, tb, children, mate, mutation)
        offs, fitnesses, enemy_lifes = evalpop(offs, env)
        pop = tb.select(pop + offs, len(pop)) #eaMuPlusLambda selection procedure



