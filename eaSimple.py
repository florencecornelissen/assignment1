import sys, os
sys.path.insert(0, 'evoman') 

from evoman.environment import Environment
from demo_controller import player_controller
from deap import tools, creator, base, algorithms
import numpy as np
from tqdm import tqdm
import pickle

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


def evaluate_pop(pop, env, toolbox):
    """
    Evaluate the whole population
    """
    t_e_i = [i for i in pop if not i.fitness.valid]
    tmp = [env for i in t_e_i]
    fits = toolbox.map(toolbox.evaluate, t_e_i, tmp)

    s = []
    solution = False
    for ind, (fit, e_l, p_l) in zip(pop, fits):
        ind.fitness.values = (fit,)
        s.append([fit, p_l-e_l])
        if e_l <= 0:
            solution = True

    i_g_i = np.argmax(s, axis=0)[1]
    i_g_v= s[i_g_i][1]

    s = [*np.mean(s, axis=0), *np.max(s, axis=0)]
    return (pop, s, solution, i_g_i, i_g_v)

def main(args):

    # array to store the run stats
    stats = np.zeros((args.n_repeats, args.gens, 4))
    won_gens = [None for _ in range(args.n_repeats)]

    for n in range(args.n_repeats):
        best_individual_gain = -10000


        # define the environment
        env = Environment(experiment_name=experiment_name,
                                            enemies=args.enemies,
                                            player_controller=player_controller(args.n_hid_neurons),
                                            speed="fastest",
                                            logs='off',
                                            level=2
                                            )

        # calculate the number of weights
        n_weights = (env.get_num_sensors() + 1) * args.n_hid_neurons + (args.n_hid_neurons+1) * 5

        creator.create('FitnessBest', base.Fitness, weights = (1.0,))
        creator.create('Individual', np.ndarray, fitness = creator.FitnessBest)

        toolbox = base.Toolbox()
        toolbox.register('indices', np.random.uniform, -1, 1) 
        toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.indices, n = n_weights)
        toolbox.register('population', tools.initRepeat, list, toolbox.individual, n = args.n_population)

        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxTwoPoint)                                  # crossover operator
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.5)     # mutation operator
        toolbox.register("select", tools.selTournament, tournsize=3)                # selection procedure

        won = False

        # initialize the population
        population = toolbox.population(n=args.n_population)
        population, stats_g, solution, i_g_i, i_g_v = evaluate_pop(population, env, toolbox)
        stats[n, 0] = stats_g
        if i_g_v > best_individual_gain:
            best_individual_gain = i_g_v
            with open(f'best_individual_eaSimple_e3_{n+1}.pickle', 'wb') as f:
                pickle.dump(population[i_g_i] , f)


        # iterate over the number of generations
        for g in tqdm(range(1, args.gens)):

            population = toolbox.select(population, len(population))
            offspring = algorithms.varAnd(population, toolbox, args.mate, args.mutation)
            offspring, stats_g, solution, i_g_i, i_g_v = evaluate_pop(offspring, env, toolbox)
            population = offspring
            if i_g_v > best_individual_gain:
                best_individual_gain = i_g_v
                with open(f'best_individual_eaSimple_e3_{n+1}.pickle', 'wb') as f:
                    pickle.dump(offspring[i_g_i] , f)


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
    parameters.enemies          = [3] 
    parameters.mutation         = 0.5
    parameters.mate             = 1
    parameters.n_hid_neurons    = 15
    parameters.n_repeats        = 9
    parameters.gens             = 50

    # run the algorithm -> stats (n_repeats X gens X 4), won_gens (n_repeats)
    stats, won_gens = main(parameters)
    with open(f'{parameters.enemies[0]}, {parameters.mutation}, {parameters.mate}, {parameters.n_hid_neurons}.npy', 'wb') as f:
        np.save(f, stats)
    print(won_gens)


