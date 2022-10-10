import sys, os
sys.path.insert(0, 'evoman') 

from evoman.environment import Environment
from demo_controller import player_controller
from deap import tools, creator, base, algorithms
import numpy as np
from tqdm import tqdm


headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'experiment-varAnd'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


def evaluate(x,env):
    """
    Evaluate a single genome by simulating a game
    """
    fitness,p_l,e_l,time_ = env.play(pcont=x)
    return fitness, e_l, p_l


def evaluate_pop(pop, envs, toolbox):
    """
    Evaluate the whole population
    """
    t_e_i = [i for i in pop if not i.fitness.valid]

    fitnesses = [[] for _ in pop]
    stats = []
    for env in envs:
        tmp = [env for i in t_e_i]
        fits = toolbox.map(toolbox.evaluate, t_e_i, tmp)

        s = []
        solution = False
        for i, (fit, e_l, p_l) in enumerate(fits):
            fitnesses[i].append(fit)
            s.append([fit, p_l-e_l])
            if e_l <= 0:
                solution = True

        stats.append([*np.mean(s, axis=0), *np.max(s, axis=0)])

    for f, ind in zip(fitnesses, pop):
        ind.fitness.values = tuple(f)
    return (pop, np.array(stats), solution)


def main(args):

    # array to store the run stats
    stats = np.zeros((args.n_repeats, args.gens, len(args.enemies), 4))
    won_gens = [None for _ in range(args.n_repeats)]

    for n in range(args.n_repeats):


        # define the environment for all enemies
        envs = [
                Environment(
                            experiment_name=experiment_name,
                            enemies=[e],
                            player_controller=player_controller(args.n_hid_neurons),
                            speed="fastest",
                            logs='off',
                            level=2,
                            enemymode='static',
                            multiplemode='no'
                            ) 
                for e in args.enemies
                ]

        # calculate the number of weights
        n_weights = (envs[0].get_num_sensors() + 1) * args.n_hid_neurons + (args.n_hid_neurons+1) * 5

        creator.create('FitnessBest', base.Fitness, weights = tuple([1.0 for _ in args.enemies]))
        creator.create('Individual', np.ndarray, fitness = creator.FitnessBest)

        toolbox = base.Toolbox()
        toolbox.register('indices', np.random.uniform, -1, 1) 
        toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.indices, n = n_weights)
        toolbox.register('population', tools.initRepeat, list, toolbox.individual, n = args.n_population)

        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxTwoPoint)                                  # crossover operator
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.5)     # mutation operator
        toolbox.register("select", args.selection)                                  # selection procedure

        won = False

        # initialize the population
        population = toolbox.population(n=args.n_population)
        population, stats_g, solution = evaluate_pop(population, envs, toolbox)
        population = toolbox.select(population, len(population))
        stats[n, 0] = stats_g


        # iterate over the number of generations
        for g in tqdm(range(1, args.gens), disable=args.silent):

            offspring = algorithms.varOr(population, toolbox, args.children, args.mate, args.mutation)
            offspring, stats_g, solution = evaluate_pop(offspring, envs, toolbox)
            population = toolbox.select(population + offspring, len(population))

            # save generation statistics
            stats[n, g] = stats_g

            if solution is True and won is False:
                won_gens[n] = g
                won = True

        del creator.Individual
        del creator.FitnessBest
    
    return stats, won_gens
    


if __name__ == "__main__":
    """
    Example run to illustrate how to run this file
    """

    # create parameters class
    parameters = type('MyClass', (object,), {'content':{}}) 

    # add parameter values
    parameters.n_population     = 100
    parameters.enemies          = [1, 2, 3] 
    parameters.mutation         = 0.1
    parameters.mate             = 0.9
    parameters.n_hid_neurons    = 10
    parameters.n_repeats        = 1
    parameters.gens             = 40
    parameters.children         = 150
    parameters.selection        = tools.selSPEA2            # tools.selNSGA2 or tools.selSPEA2
    parameters.silent           = False


    # run the algorithm -> stats (n_repeats X gens X 4), won_gens (n_repeats)
    stats, won_gens = main(parameters)


    # DEPRICATED

    # print(stats)
    # print(won_gens)
    
    # get the fitness
    # fitness = stats[:, :, :, 0]
    # mean_fitness = np.mean(fitness, axis=0)
    # std_fitness = np.std(fitness, axis=0)

    # # calculate the individual gain, i.e. (player_health - enemy_health)
    # individual_gain = stats[:, :, :, 1]
    # mean_individual_gain = np.mean(individual_gain, axis=0)
    # std_individual_gain = np.std(individual_gain, axis=0)

    # # maximum fitness
    # max_fitness = stats[:, :, :, 2]
    # mean_max_fitness = np.mean(max_fitness, axis=0)
    # std_max_fitness = np.std(max_fitness, axis=0)

    # # maximum individual gain
    # max_ind_gain = stats[:, :, :, 3]
    # mean_max_ind_gain = np.mean(max_ind_gain, axis=0)
    # std_max_ind_gain = np.std(max_ind_gain, axis=0)

    # # calculate the number of iterations until a solution was found
    # gens_until_solution = won_gens
    # try:
    #     mean_gens_until_solution = np.nanmean(gens_until_solution, axis=0)
    # except:
    #     mean_gens_until_solution = None

