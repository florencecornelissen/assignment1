from eaSimple import *


for mutation in [0.1, 0.2, 0.3, 0.5]:
    for mate in [0.1, 0.2, 0.3, 0.5]:
        n_hid_neurons = 10

        parameters = type('MyClass', (object,), {'content':{}}) 
        parameters.enemies          = [4] 
        parameters.n_population     = 100
        parameters.n_repeats        = 3
        parameters.gens             = 35
        
        parameters.mutation = mutation
        parameters.mate = mate
        parameters.n_hid_neurons = n_hid_neurons

        try:
            stats, won_gens = main(parameters)
            fitness = stats[:, :, 0]
            individual_gain = stats[:, :, 1]
            mean_individual_gain = np.mean(individual_gain, axis=0)
            mean_fitness = np.mean(fitness, axis=0)
            s1 = f'{won_gens}, {mean_fitness[-1]}, {mean_individual_gain[-1]}\n'
            s2 = f' {mutation}, {mate}, {n_hid_neurons}, \n\n\n'

            f = open('results.txt', 'a')
            f.write(s1)
            f.write(s2)
            f.close()
        except:
            print('parameter space failed for some reason, continueing to the next one.. :(')
            