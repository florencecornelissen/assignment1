import random

"""
Dingen die jullie kunnen aanpassen:
- het aantal hyperparameters h1, h2,..., hn; en hun ranges
- het aantal configuraties n
"""

# example ranges for 3 hyperparameters
h1 = [0.1 , 0.2, 0.3, 0.4]
h2 = [0.0001, 0.001, 0.01, 0.1]
h3 = [1 , 20 , 100]

# save all possible configurations
configurations = []
for i in h1:
    for j in h2:
        for k in h3:
            configurations.append((i, j, k))

# take n configurations randomly
n = 20 
random_sample = random.sample(configurations, n)

# split sample in roughly 5 equal parts
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
random_sample_divided = list(split(random_sample, 5))

# list of hyperparameter configurations each person has to test
config_elise = random_sample_divided[0]
config_lisa = random_sample_divided[1]
config_flo = random_sample_divided[2]
config_casper = random_sample_divided[3]
config_samantha = random_sample_divided[4]

# save in file
with open('hyperparam_configurations.txt', 'w') as f:
    f.write('elise: ' + str(config_elise) + '\n')
    f.write('lisa: ' + str(config_lisa) + '\n')
    f.write('flo: ' + str(config_flo) + '\n')
    f.write('casper: ' + str(config_casper) + '\n')
    f.write('samantha: ' + str(config_samantha) + '\n')