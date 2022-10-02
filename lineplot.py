import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

def get_data(stats):
    """
    Function that returns fitness data of a certain EA against a certain enemy
    """

    # get the fitness
    fitness = stats[:, :, 0]
    mean_fitness = np.mean(fitness, axis=0)
    std_fitness = np.std(fitness, axis=0)
    # maximum fitness
    max_fitness = stats[:, :, 2]
    mean_max_fitness = np.mean(max_fitness, axis=0)
    std_max_fitness = np.std(max_fitness, axis=0)

    return mean_fitness, std_fitness, mean_max_fitness, std_max_fitness


def line_plot(enemy, mean_f_simple, max_f_simple, std_mean_simple, std_max_simple,
              mean_f_lambda, max_f_lambda, std_mean_lambda, std_max_lambda):
    """
    Function that returns lineplot
    Param:  enemy: 1, 2 or 3
            mean_f_x: mean fitness of algorithm x
            max_f_x: max fitness of algorithm x
            std_mean_x: std mean fitness of algorithm x
            std_max_x: stf max fitness of algorithm x
    """
    title = 'Enemy ' + str(enemy)
    title_2 = 'lineplot_enemy' + str(enemy) + '.png'

    generations = [i for i in range(len(mean_f_simple))]

    plt.errorbar(generations, mean_f_simple, std_mean_simple, fmt='-o', ms=3, capsize=3, color='black', label='eaSimple: average mean')
    plt.errorbar(generations, max_f_simple, std_max_simple, fmt='-o', ms=3, capsize=3, label='eaSimple: average max')
    plt.errorbar(generations, mean_f_lambda, std_mean_lambda, fmt='-o', ms=3, capsize=3, label='eaMuPlusLambda: average mean')
    plt.errorbar(generations, max_f_lambda, std_max_lambda, fmt='-o', ms=3, capsize=3, label='eaMuPlusLambda: average max')
    plt.xlabel('generations')
    plt.ylabel('fitness')
    plt.title(title)
    plt.legend()
    plt.savefig(title_2, bbox_inches='tight')
    plt.show()

# # example
# mean_f_simple = [1, 2, 3, 4, 5, 5.5, 5.75, 6, 6]
# std_mean_simple = [0.7 for i in range(9)]
# max_f_simple = [1, 3, 5, 6, 7, 8, 8.5, 9, 9]
# std_max_simple = [0.7 for i in range(9)]
# mean_f_lambda = [1, 4, 7, 10, 12, 13, 14, 14, 14]
# std_mean_lambda = [0.8 for i in range(9)]
# max_f_lambda = [1, 5, 9, 11, 13, 14, 15, 15, 15]
# std_max_lambda = [0.8 for i in range(9)]
# line_plot(1, mean_f_simple, max_f_simple, std_mean_simple, std_max_simple,
#               mean_f_lambda, max_f_lambda, std_mean_lambda, std_max_lambda)


# # once we have all the files
# with open('best_eaSimple_e3.npy', 'rb') as f:
#     stats_eaSimple_e3 = np.load(f)
with open('best_eaSimple_e4.npy', 'rb') as f:
    stats_eaSimple_e4 = np.load(f)
with open('best_eaSimple_e7.npy', 'rb') as f:
    stats_eaSimple_e7 = np.load(f)
# with open('best_eaMuPlusLambda_e3.npy', 'rb') as f:
#     stats_eaMuPlusLambda_e3 = np.load(f)
with open('best_eaMuPlusLambda_e4.npy', 'rb') as f:
    stats_eaMuPlusLambda_e4 = np.load(f)
with open('best_eaMuPlusLambda_e7.npy', 'rb') as f:
    stats_eaMuPlusLambda_e7 = np.load(f)

# enemy 3
# mean_fitness_simple, std_fitness_simple, mean_max_fitness_simple, std_max_fitness_simple = get_data(stats_eaSimple_e3)
# mean_fitness_lambda, std_fitness_lambda, mean_max_fitness_lambda, std_max_fitness_lambda = get_data(stats_eaMuPlusLambda_e3)
# line_plot(3, mean_fitness_simple, mean_max_fitness_simple, std_fitness_simple, std_max_fitness_simple,
#               mean_fitness_lambda, mean_max_fitness_lambda, std_fitness_lambda, std_max_fitness_lambda)

# # enemy 4
mean_fitness_simple, std_fitness_simple, mean_max_fitness_simple, std_max_fitness_simple = get_data(stats_eaSimple_e4)
mean_fitness_lambda, std_fitness_lambda, mean_max_fitness_lambda, std_max_fitness_lambda = get_data(stats_eaMuPlusLambda_e4)
line_plot(4, mean_fitness_simple, mean_max_fitness_simple, std_fitness_simple, std_max_fitness_simple,
              mean_fitness_lambda, mean_max_fitness_lambda, std_fitness_lambda, std_max_fitness_lambda)

# enemy 7
mean_fitness_simple, std_fitness_simple, mean_max_fitness_simple, std_max_fitness_simple = get_data(stats_eaSimple_e7)
mean_fitness_lambda, std_fitness_lambda, mean_max_fitness_lambda, std_max_fitness_lambda = get_data(stats_eaMuPlusLambda_e7)
# line_plot(7, mean_fitness_simple, mean_max_fitness_simple, std_fitness_simple, std_max_fitness_simple,
#               mean_fitness_lambda, mean_max_fitness_lambda, std_fitness_lambda, std_max_fitness_lambda)