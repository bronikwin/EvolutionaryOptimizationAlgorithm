# -*- coding: utf-8 -*-
"""
Created by 'kolya' on 01.06.2022 at 1:38
"""
import numpy as np
""" 
pack of math functions classes to test your optimization algorithms 

you can use root_2D to plot these roots to contour map or use root_3D to plot
these roots to surface map
"""

class Himmelblau:
    def calc(self, x_grid, y_grid):
        function = (x_grid ** 2 + y_grid - 11) ** 2 + (x_grid + y_grid ** 2 - 7) ** 2
        return function

    def root_2D(self):
        return [[3, 2], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584428, -1.848126]]

    def root_3D(self):
        return [[3, 2, 0], [-2.805118, 3.131312, 0], [-3.779310, -3.283186, 0], [3.584428, -1.848126, 0]]


class Eggholder:
    def calc(self, x_grid, y_grid):
        function = -(y_grid + 47) * np.sin((abs(x_grid / 2 + (y_grid + 47)))) - x_grid * np.sin(
            (abs(x_grid - y_grid - 47)))
        return function

    def root_2D(self):
        return [[512, 404.2319]]

    def root_3D(self):
        return [[512, 404.2319, -959.6407]]


class Rosenbrok:
    def calc(self, x_grid, y_grid):
        function = 100 * (y_grid - x_grid ** 2) ** 2 + (1 - x_grid) ** 2
        return function

    def root_2D(self):
        return [[1, 0]]

    def root_3D(self):
        return [[1, 0, 0]]


class Rastrigin:
    def calc(self, x_grid, y_grid):
        function = 0.1 * x_grid ** 2 + 0.1 * y_grid ** 2 - 4 * np.cos(0.8 * x_grid) - 4 * np.cos(0.8 * y_grid) + 8
        return function

    def root_2D(self):
        return [[0, 0]]

    def root_3D(self):
        return [[0, 0,0]]


class Sombrero:
    def calc(self, x_grid, y_grid):
        function = (1 - np.sin((x_grid ** 2 + y_grid ** 2) ** 0.5) ** 2) / (1 + 0.001 * (x_grid ** 2 + y_grid ** 2))
        return function

    def root_2D(self):
        """max"""
        return [[0, 0]]

    def root_3D(self):
        """max"""
        return [[0, 0, 1]]


class Griewank:
    def calc(self, x_grid, y_grid):
        function = -10 / (0.005 * (x_grid ** 2 + y_grid ** 2) - np.cos(x_grid) * np.cos(y_grid / 2 ** 0.5) + 2) + 10
        return function

    def root_2D(self):
        return [[0, 0]]

    def root_3D(self):
        return [[0, 0,0]]


class De_jong_2:
    def calc(self, x_grid, y_grid):
        function = -100 / (100 * (x_grid ** 2 - y_grid) + (1 - x_grid) ** 2 + 1) + 100
        return function

    def root_2D(self):
        return [[0, 0]]

    def root_3D(self):
        return [[0, 0, -1]]


class Multiextremal4:
    def calc(self, x_grid, y_grid):
        function = -0.5 * (x_grid ** 2 + x_grid * y_grid + y_grid ** 2) * (
                    1 + 0.5 * np.cos(1.5 * x_grid) * np.cos(3.2 * x_grid * y_grid) * np.cos(3.14 * y_grid) +
                    0.5 * np.cos(2.2 * x_grid) * np.cos(4.8 * x_grid * y_grid) * np.cos(3.5 * y_grid))
        return function

    def root_2D(self):
        return [[0, 0]]

    def root_3D(self):
        return [[0, 0,0]]


class Rana_function:
    def calc(self, x_grid, y_grid):
        function = x_grid * np.sin((abs(y_grid + 1 - x_grid)) ** 0.5) * np.cos((abs(y_grid + 1 + x_grid)) ** 0.5) + (
                    y_grid + 1) * np.cos((abs(y_grid + 1 - x_grid)) ** 0.5) * np.sin((abs(y_grid + 1 + x_grid)) ** 0.5)

        return function

    def root_2D(self):
        return [[-488.6326, 512]]

    def root_3D(self):
        return [[-488.6326, 512, -511.7328819]]

class Bill_function:
    def calc(self, x_grid, y_grid):
        function = (1.5 - x_grid + x_grid * y_grid) ** 2 + (2.25 - x_grid + x_grid * y_grid ** 2) ** 2 + (
                    2.625 - x_grid + x_grid * y_grid ** 3) ** 2

        return function

    def root_2D(self):
        return [[3, 0.5]]

    def root_3D(self):
        return [[3, 0.5, 0]]

class Bukin_6_function:
    def calc(self, x_grid, y_grid):
        function = 100 * (abs(y_grid - 0.01 * x_grid ** 2)) ** 0.5 + 0.01 * abs(x_grid + 10)

        return function

    def root_2D(self):
        return [[-10, 1]]

    def root_3D(self):
        return [[-10, 1,0]]

class Izom_function:
    def calc(self, x_grid, y_grid):
        function = -np.cos(x_grid) * np.cos(y_grid) * np.exp(-((x_grid - np.pi) ** 2 + (y_grid - np.pi) ** 2))

        return function

    def root_2D(self):
        return [[np.pi, np.pi]]

    def root_3D(self):
        return [[np.pi, np.pi, -1]]

class Cross_tray_function:
    def calc(self, x_grid, y_grid):
        function = -0.0001 * (abs(np.sin(x_grid) * np.sin(y_grid) * np.exp(
            abs(100 - (x_grid ** 2 + y_grid ** 2) ** 0.5 / np.pi))) + 1) * 0.1

        return function

    def root_2D(self):
        return [[1.34941, -1.34941], [1.34941, 1.34941], [-1.34941, 1.34941], [-1.34941, -1.34941]]

    def root_3D(self):
        return [[1.34941, -1.34941, -2.06261], [1.34941, 1.34941, -2.06261], [-1.34941, 1.34941, -2.06261], [-1.34941, -1.34941, -2.06261]]

class Shaffer_2:
    def calc(self, x_grid, y_grid):
        function = 0.5 + ((np.sin(x_grid ** 2 - y_grid ** 2)) ** 2 - 0.5) / (
                    1 + 0.001 * (x_grid ** 2 + y_grid ** 2)) ** 2

        return function

    def root_2D(self):
        return [[0, 0]]

    def root_3D(self):
        return [[0, 0,0]]

class Shaffer_4:
    def calc(self, x_grid, y_grid):
        function = 0.5 + (np.cos(np.sin(abs(x_grid ** 2 - y_grid ** 2))) ** 2 - 0.5) / (
                    1 + 0.001 * (x_grid ** 2 + y_grid ** 2)) ** 2

        return function

    def root_2D(self):
        return [[0, 1.25313]]

    def root_3D(self):
        return [[0, 1.25313, 0.292579]]

class Holder_function:
    def calc(self, x_grid, y_grid):
        function = -abs(np.sin(x_grid) * np.cos(y_grid) * np.exp(abs(1 - (x_grid ** 2 + y_grid ** 2) ** 0.5 / np.pi)))

        return function

    def root_2D(self):
        return [[8.05502, 9.66459],[-8.05502, 9.66459],[8.05502, -9.66459],[-8.05502, -9.66459]]

    def root_3D(self):
        return [[8.05502, 9.66459,-19.2085],[-8.05502, 9.66459,-19.2085],[8.05502, -9.66459,-19.2085],[-8.05502, -9.66459,-19.2085]]

class Levi_13_function:
    def calc(self, x_grid, y_grid):
        function = np.sin(3 * np.pi * x_grid) ** 2 + (x_grid - 1) ** 2 * (1 + np.sin(3 * np.pi * y_grid) ** 2) + (
                    y_grid - 1) ** 2 * (1 + np.sin(2 * np.pi * y_grid) ** 2)

        return function

    def root_2D(self):
        return [[1, 1]]

    def root_3D(self):
        return [[1, 1,0]]

class Ecli_function:
    def calc(self, x_grid, y_grid):
        function = -20 * np.exp(-0.2 * (0.5 * (x_grid ** 2 + y_grid ** 2)) ** 0.5) - np.exp(
            0.5 * np.cos(2 * np.pi * x_grid) + np.cos(2 * np.pi * y_grid)) + np.e + 20

        return function

    def root_2D(self):
        return [[0, 0]]

    def root_3D(self):
        return [[0, 0,0]]

class Matias_function:
    def calc(self, x_grid, y_grid):
        function = 0.26 * (x_grid ** 2 + y_grid ** 2) - 0.48 * x_grid * y_grid

        return function

    def root_2D(self):
        return [[0, 0]]

    def root_3D(self):
        return [[0, 0,0]]

class Goldshein_price:
    def calc(self, x_grid, y_grid):
        function = (1 + (x_grid + y_grid + 1) ** 2 * (
                    19 - 14 * x_grid + 3 * x_grid ** 2 - 14 * y_grid + 6 * x_grid * y_grid + 3 * y_grid ** 2)) * (
                               30 + (2 * x_grid - 3 * y_grid) ** 2 * (
                                   18 - 32 * x_grid + 12 * x_grid ** 2 + 48 * y_grid - 36 * x_grid * y_grid + 27 * y_grid ** 2))

        return function

    def root_2D(self):
        return [[0, -1]]

    def root_3D(self):
        return [[0, -1,3]]

class Katnikov:
    def calc(self, x_grid, y_grid):
        function = 0.5 * (x_grid ** 2 + y_grid ** 2) * (
                    2 * 0.8 + 0.8 * np.cos(x_grid * 1.5) * np.cos(y_grid * 3.14) + 0.8 * np.cos(
                5 ** 0.5 * x_grid) * np.cos(3.5 * y_grid))

        return function

    def root_2D(self):
        return [[0, 0]]

    def root_3D(self):
        return [[0, 0,0]]



if __name__ == '__main__':
    pass
