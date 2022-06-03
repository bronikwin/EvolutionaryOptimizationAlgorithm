# -*- coding: utf-8 -*-
"""
Created by 'kolya' on 03.06.2022 at 11:18
"""
from evolutionary_optimization_deap import *

if __name__ == '__main__':
    """ Выбирай любую функцию из OPT """
    FUNCTION = OPT.Griewank()
    POPULATION_SIZE = 200
    P_CROSSOVER = 0.95
    P_MUTATION = 0.1
    MAX_GENERATIONS = 30
    HALL_OF_FAME_SIZE = 3
    RANDOM_SEED = 42

    LOW, UP = -16, 16
    STEP = 1
    ETA = 20

    FIND_MIN = True
    PLOT_AS_SURFACE = True

    evolutionary_optimization(FUNCTION, POPULATION_SIZE, P_CROSSOVER, P_MUTATION, MAX_GENERATIONS, HALL_OF_FAME_SIZE,
                              ETA, LOW, UP, STEP, FIND_MIN=FIND_MIN,
                              RANDOM_SEED=RANDOM_SEED, PLOT_AS_SURFACE=PLOT_AS_SURFACE, )