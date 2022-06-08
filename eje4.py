import array
import random
import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

puntos = ['a', 'b', 'c', 'd', 'e']

#    a  b  c  d  e
# a   0  7  9  8  20
# b   7  0  10 4  11
# c   9  10 0  15 5
# d   8  4  15 0  17
# e   20 11 5  17 0

distancias = [
    [0, 7, 9, 8, 20],
    [7, 0, 10, 4, 11],
    [9, 10, 0, 15, 5],
    [8, 4, 15, 0, 17],
    [20, 11, 5, 17, 0]]


distance_map = distancias
IND_SIZE = 5

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode='i',
               fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("indices", random.sample, range(IND_SIZE), IND_SIZE)

# Structure initializers
toolbox.register("individual", tools.initIterate,
                 creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evalTSP(individual):
    distance = distance_map[individual[-1]][individual[0]]
    for gene1, gene2 in zip(individual[0:-1], individual[1:]):
        distance += distance_map[gene1][gene2]
    return distance,


toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evalTSP)


def main():
    random.seed(169)

    pop = toolbox.population(n=2)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    algorithms.eaSimple(pop, toolbox, 0.7, 0.2, 40, stats=stats,
                        halloffame=hof)

    return pop, stats, hof
    
if __name__ == "__main__":
    main()