from collections import Counter
from typing import List


def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    return Counter(x) == Counter(y)


def max_prod_mod_3(x: List[int]) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    max_product = -1
    for i in range(len(x) - 1):
        prod = x[i] * x[i + 1]
        if x[i] % 3 == 0 or x[i + 1] % 3 == 0:
            max_product = max(max_product, prod)
    return max_product


def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    """
    Сложить каналы изображения с указанными весами.
    """
    height = len(image)
    width = len(image[0])
    num_channels = len(image[0][0])
    
    # Проверить, что длина вектора весов совпадает с количеством каналов
    assert num_channels == len(weights), "Длина вектора весов должна совпадать с количеством каналов изображения"
    
    # Инициализировать результирующую матрицу
    result = [[0.0 for _ in range(width)] for _ in range(height)]
    
    # Пройтись по каждому пикселю изображения
    for i in range(height):
        for j in range(width):
            # Сложить взвешенные значения каналов для каждого пикселя
            result[i][j] = sum(image[i][j][k] * weights[k] for k in range(num_channels))
    
    return result


def rle_scalar(x: List[List[int]], y:  List[List[int]]) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """

    def decode_rle(rle: List[List[int]]) -> List[int]:
        """
        Раскодировать вектор из формата RLE в обычный вектор.
        """
        decoded = []
        for value, count in rle:
            decoded.extend([value] * count)
        return decoded

    # Раскодировать оба вектора
    decoded_x = decode_rle(x)
    decoded_y = decode_rle(y)

    # Проверить, совпадают ли длины раскодированных векторов
    if len(decoded_x) != len(decoded_y):
        return -1

    # Вычислить скалярное произведение
    scalar_product = sum(a * b for a, b in zip(decoded_x, decoded_y))
    
    return scalar_product


def cosine_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y. 
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    res = [[0] * len(Y) for _ in range(len(X))]
    for i in range(len(X)):
        for j in range(len(Y)):
            normx = 0
            normy = 0
            for d in range(len(X[0])):
                normx += X[i][d]**2
                normy += Y[j][d]**2
            normx = normx**0.5
            normy = normy**0.5
            if normx == 0 or normy == 0:
                res[i][j] = 1
            else:
                for d in range(len(X[0])):
                    res[i][j] += X[i][d] * Y[j][d]
                res[i][j] /= normy
                res[i][j] /= normx
    return res