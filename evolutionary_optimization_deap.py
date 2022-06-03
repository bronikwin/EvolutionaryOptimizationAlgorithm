# -*- coding: utf-8 -*-
"""
Created by 'kolya' on 02.06.2022 at 23:03
"""
import random
import time
import numpy as np
import math_functions as OPT
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import matplotlib.pyplot as plt

def evolutionary_optimization(FUNCTION,
                               POPULATION_SIZE=100,
                               P_CROSSOVER=0.9,
                               P_MUTATION=0.1,
                               MAX_GENERATIONS=50,
                               HALL_OF_FAME_SIZE=3,
                               ETA=0.2,
                               LOW=-5,
                               UP=5,
                              STEP=1,
                              FIND_MIN=True,
                              RANDOM_SEED=42,
                              PLOT_AS_SURFACE=True):
    """
    :param FUNCTION: оптимизируемая функция
    :param POPULATION_SIZE: размер популяции (число особей)
    :param P_CROSSOVER: вероятность скрещивания
    :param P_MUTATION: вероятность мутации
    :param MAX_GENERATIONS: макс. количество поколений
    :param HALL_OF_FAME_SIZE: зал славы
    :param RANDOM_SEED: зерно рандома
    :param ETA: схожесть особей
    :param LOW: нижняя граница поиска максимума/минимума функции
    :param UP: верхняя граница поиска максимума/минимума функции
    :param STEP: шаг отображения графика/поверхности (НЕ ВЛИЯЕТ НА РЕЗУЛЬТАТ, но влияет на корректное отображение графика.
    :param MIN: if True, -> weights==-1 (minimum of the function), else weights==1 (maximum of the function)
    :param PLOT_AS_SURFACE: if True -> PLOT IS SURFACE, ELSE PLOT IS CONTOUR MAP
    :return:
    """
    LENGTH_CHROM=2 # хромосома состоит из 2 значений: x,y
    def randomPoint(a, b):
        """ Функция создания начальной точки в диапазоне от a до b """
        return [random.uniform(a, b), random.uniform(a, b)]

    def make_x_y_grid(xmin: int, xmax: int, ymin: int, ymax: int, step: float = 1) -> tuple:
        """ для создания meshgrid """
        x = np.arange(xmin, xmax, step)
        y = np.arange(ymin, ymax, step)
        x_grid, y_grid = np.meshgrid(x, y)
        return (x_grid, y_grid)

    def func_wrapper(individual) -> tuple:
        """ обертка под класс тестовых функций """
        x, y = individual
        return FUNCTION.calc(x, y),

    def show_evo_algorithm(ax,x,y,function, population=[], roots=[], LOW=1, UP=1, PLOT_AS_SURFACE=True,):
        """ функция для визуализации процесса поиска решения """
        ax.clear()
        if PLOT_AS_SURFACE:
            ax.plot_surface(x, y, function, cmap='gist_rainbow',alpha=0.5)

            population=[[a,b,func_wrapper([a,b])] for a,b in population]
        else:
            ax.contour(x,y,function)
        ax.scatter(*zip(*roots), color='red', alpha=1)
        ax.scatter(*zip(*population), color='green', alpha=1)

        plt.xlim(LOW-1,UP+1)
        plt.ylim(LOW-1,UP+1)
        plt.draw()
        plt.gcf().canvas.flush_events()
        time.sleep(0.2)

    random.seed(RANDOM_SEED)
    hof=tools.HallOfFame(HALL_OF_FAME_SIZE)

    if FIND_MIN:
        weight=-1.0
    else:
        weight=1.0
    creator.create("Fitness", base.Fitness, weights=(weight,)) # определение функции приспособленности
    creator.create("Individual", list, fitness=creator.Fitness) # создание инвидуумов


    toolbox=base.Toolbox()
    toolbox.register("randomPoint", randomPoint, LOW, UP) # регистрация функции randomPoint
    toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.randomPoint) # регистрация функции individualCreator
    toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator) # регистрация функции populationCreator

    population=toolbox.populationCreator(n=POPULATION_SIZE)

    toolbox.register('evaluate', func_wrapper) # функция приспособленности
    toolbox.register('select', tools.selTournament, tournsize=3) # турнирный отбор особей
    toolbox.register('mate', tools.cxSimulatedBinaryBounded, low=LOW,up=UP,eta=ETA) # функция скрещивания
    toolbox.register('mutate', tools.mutPolynomialBounded, low=LOW,up=UP,eta=ETA, indpb=1.0/LENGTH_CHROM) # функция мутации. indpb - вероятность мутации отдельного гена в хромосоме

    stats=tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min",np.min)
    stats.register("avg",np.mean)

    # -----------------------------------------------------
    x_grid,y_grid=make_x_y_grid(LOW, UP, LOW, UP, step=STEP)
    function=FUNCTION.calc(x_grid, y_grid)

    # В зав-ти от PLOT_AS_SURFACE строится как контур или как поверхность.
    if PLOT_AS_SURFACE:roots=FUNCTION.root_3D()
    else: roots=FUNCTION.root_2D()

    plt.ion()
    if PLOT_AS_SURFACE:ax = plt.axes(projection='3d')
    else: ax=plt.axes()
    population, logbook=algorithms.eaSimple(population, toolbox,
                                            cxpb=P_CROSSOVER,
                                            mutpb=P_MUTATION ,
                                            ngen=MAX_GENERATIONS,
                                            halloffame=hof,
                                            stats=stats,
                                            verbose=True,
                                            callback=(show_evo_algorithm, (ax,x_grid,y_grid,function, population, roots, LOW, UP, PLOT_AS_SURFACE)))

    maxFitnessValue, meanFitnessValue=logbook.select('min','avg')

    best=hof.items[0]
    print('Найденное решение (x,y):',best)
    print('Известное решение (x,y):', FUNCTION.root_2D())

    plt.ioff()
    plt.show()

    plt.plot(maxFitnessValue, color='red')
    plt.plot(meanFitnessValue, color='green')
    plt.show()

if __name__ == '__main__':
    FUNCTION = OPT.Levi_13_function()
    POPULATION_SIZE = 200
    P_CROSSOVER = 0.9
    P_MUTATION = 0.1
    MAX_GENERATIONS = 30
    HALL_OF_FAME_SIZE = 3
    RANDOM_SEED = 42

    LOW, UP = -16, 16
    STEP = 2
    ETA = 20

    FIND_MIN=True
    PLOT_AS_SURFACE=True

    evolutionary_optimization(FUNCTION, POPULATION_SIZE, P_CROSSOVER, P_MUTATION, MAX_GENERATIONS, HALL_OF_FAME_SIZE, ETA, LOW, UP, STEP, FIND_MIN=FIND_MIN,
                              RANDOM_SEED=RANDOM_SEED, PLOT_AS_SURFACE=PLOT_AS_SURFACE,)