from typing import List
from copy import deepcopy


def get_part_of_array(X: List[List[float]]) -> List[List[float]]:
    """
    X - двумерный массив вещественных чисел размера n x m. Гарантируется что m >= 500
    Вернуть: двумерный массив, состоящий из каждого 4го элемента по оси размерности n 
    и c 120 по 500 c шагом 5 по оси размерности m
    """
     # Получаем количество строк и столбцов
    n = len(X)
    m = len(X[0])

    # Инициализируем результат
    result = []

    # Сначала добавим каждую 4-ю строку в результат
    for i in range(0, n, 4):  # каждый 4-й элемент по оси n
        row = []
        for j in range(120, 500, 5):  # от 120 до 500 с шагом 5 по оси m
            row.append(X[i][j])
        result.append(row)

    return result



def sum_non_neg_diag(X: List[List[int]]) -> int:
    """
    Вернуть  сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """
    n = len(X)
    m = len(X[0])
    sum_diag = 0
    has_non_neg = False

    for i in range(min(n, m)):
        if X[i][i] >= 0:
            sum_diag += X[i][i]
            has_non_neg = True

    return sum_diag if has_non_neg else -1


def replace_values(X: List[List[float]]) -> List[List[float]]:
    """
    X - двумерный массив вещественных чисел размера n x m.
    По каждому столбцу нужно почитать среднее значение M.
    В каждом столбце отдельно заменить: значения, которые < 0.25M или > 1.5M на -1
    Вернуть: двумерный массив, копию от X, с измененными значениями по правилу выше
    """
    # Создаем копию массива
    X_copy = deepcopy(X)
    n = len(X)
    m = len(X[0])

    for j in range(m):
        # Вычисляем среднее значение для столбца
        col_sum = sum(X[i][j] for i in range(n))
        col_mean = col_sum / n

        # Заменяем значения в копии массива
        for i in range(n):
            if X[i][j] < 0.25 * col_mean or X[i][j] > 1.5 * col_mean:
                X_copy[i][j] = -1

    return X_copy
