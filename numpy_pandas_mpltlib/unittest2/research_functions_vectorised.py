import numpy as np


def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    unique_x, counts_x = np.unique(x, return_counts=True)
    unique_y, counts_y = np.unique(y, return_counts=True)
    
    dict_x = dict(zip(unique_x, counts_x))
    dict_y = dict(zip(unique_y, counts_y))
    
    return dict_x == dict_y


def max_prod_mod_3(x: np.ndarray) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    if len(x) < 2:
        return -1

    # Вычислить произведения соседних элементов
    products = x[:-1] * x[1:]

    # Найти индексы, где хотя бы один элемент делится на 3
    mask = (x[:-1] % 3 == 0) | (x[1:] % 3 == 0)

    # Отфильтровать произведения по маске
    filtered_products = products[mask]

    # Если таких произведений нет, вернуть -1
    if len(filtered_products) == 0:
        return -1

    # Вернуть максимальное произведение
    return np.max(filtered_products)


def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Сложить каналы изображения с указанными весами.
    """
    # Преобразовать вектор весов в numpy массив
    weights_np = np.array(weights)
    
    # Проверить, что длина вектора весов совпадает с количеством каналов
    assert image.shape[2] == len(weights_np), "Длина вектора весов должна совпадать с количеством каналов изображения"
    
    # Умножить каждый канал на соответствующий вес и сложить каналы
    result = np.tensordot(image, weights_np, axes=([2], [0]))
    
    return result


def rle_scalar(x: np.ndarray, y: np.ndarray) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    if x[:, 1].sum() != y[:, 1].sum():
        return -1
    newx = np.repeat(x[:, 0], x[:, 1])
    newy = np.repeat(y[:, 0], y[:, 1])
    result = np.dot(newx, newy)
    return result


def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y.
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    norm_X = np.linalg.norm(X, axis=1, keepdims=True)
    norm_Y = np.linalg.norm(Y, axis=1, keepdims=True)
    norms = norm_X.dot(norm_Y.T)
    res = np.dot(X, Y.T)
    
    cosine_distance = np.where(norms == 0, 1, res / norms)
    
    return cosine_distance