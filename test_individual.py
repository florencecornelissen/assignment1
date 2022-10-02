import sys, os
sys.path.insert(0, 'evoman') 

import pickle
from deap import creator, base
import numpy as np
from evoman.environment import Environment
from demo_controller import player_controller
import glob
import matplotlib.pyplot as plt
from scipy import stats as st
import seaborn as sns

sns.set_theme()

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'experiment-varAnd'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


creator.create('FitnessBest', base.Fitness, weights = (1.0,))
creator.create('Individual', np.ndarray, fitness = creator.FitnessBest)


fig, axes = plt.subplots(1, 3, sharey=True)

for i, enemy in enumerate([3, 4, 7]):
    env = Environment(experiment_name=experiment_name,
                                    enemies=[enemy],
                                    player_controller=player_controller(15),
                                    speed="fastest",
                                    logs='off',
                                    level=2
                                    )

    ig_eaSimple = []
    for filepath in glob.iglob(f'best_individual/pickle/eaSimple_{enemy}/*'):
        individual = pickle.load(open(filepath, 'rb'))
        fitness, p_l, e_l, time_ = env.play(pcont=individual)
        ig_eaSimple.append(p_l-e_l)


    ig_eaMuPlusLambda = []
    for filepath in glob.iglob(f'best_individual/pickle/eaMuPlusLambda_{enemy}/*'):
        individual = pickle.load(open(filepath, 'rb'))
        fitness, p_l, e_l, time_ = env.play(pcont=individual)
        ig_eaMuPlusLambda.append(p_l-e_l)
    

    print(st.ttest_ind(a=ig_eaMuPlusLambda, b=ig_eaSimple, equal_var=True, alternative='greater'))

    data = ig_eaSimple, ig_eaMuPlusLambda
    title = 'Enemy ' + str(enemy)
    x = np.repeat(['1', '2'], 10)
    g = sns.boxplot(x=x, y=np.array(data).flatten(), width=0.4, palette='Set2', ax=axes[i])
    axes[i].set_title(f'enemy {enemy}')
    axes[i].set_xticks([])

plt.ylim([50, 95])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
axes[0].set_ylabel('Mean individual gain')
plt.savefig('boxplot.png', bbox_inches='tight')