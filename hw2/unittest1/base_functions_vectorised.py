import numpy as np

def get_part_of_array(X: np.ndarray) -> np.ndarray:
    """
    X - двумерный массив размера n x m. Гарантируется что m >= 500
    Вернуть: двумерный массив, состоящий из каждого 4го элемента по оси размерности n 
    и c 120 по 500 c шагом 5 по оси размерности m
    """
    # Преобразуем X в numpy массив, если это не так
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    
    # Извлекаем нужные элементы с использованием срезов и шагов
    result = X[::4, 120:500:5]

    return result


def sum_non_neg_diag(X: np.ndarray) -> int:
    """
    Вернуть  сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """
    # Преобразуем X в numpy массив, если это не так
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    # Извлекаем главную диагональ
    diag = np.diag(X)
    # Фильтруем неотрицательные элементы
    non_neg_diag = diag[diag >= 0]
    
    # Если нет неотрицательных элементов, возвращаем -1
    if non_neg_diag.size == 0:
        return -1
    
    return np.sum(non_neg_diag)


def replace_values(X: np.ndarray) -> np.ndarray:
    """
    X - двумерный массив вещественных чисел размера n x m.
    По каждому столбцу нужно почитать среднее значение M.
    В каждом столбце отдельно заменить: значения, которые < 0.25M или > 1.5M на -1
    Вернуть: двумерный массив, копию от X, с измененными значениями по правилу выше
    """
    # Преобразуем X в numpy массив, если это не так
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    X_copy = X.copy()

    col_means = np.mean(X, axis=0)
    
    # Создаем маски для значений, которые нужно заменить
    mask_low = X < 0.25 * col_means
    mask_high = X > 1.5 * col_means
    
    # Применяем маски к копии массива
    X_copy[mask_low | mask_high] = -1
    
    
    return X_copy
