import numpy as np
import typing

# Признаки книг: [техничность, магичность, романтика]
books = {
    'Грокаем алгоритмы': np.array([9, 1, 0]),
    'Ведьмак': np.array([1, 9, 3]),
    '1984': np.array([5, 2, 1]),
    'Гарри Поттер': np.array([2, 8, 4]),
    'Математика для Data Science': np.array([10, 0, 0])
}


def cosine_similarity(vec_a: typing.List[float], vec_b: typing.List[float]) -> float:
    """Косинусное произведение двух векторов."""

    if len(vec_a) != len(vec_b):
        raise ValueError("Векторы должны быть равной длины")

    val_mult = 0

    for a, b in zip(vec_a, vec_b):
        val_mult += a * b

    len_a = np.sqrt(sum(a**2 for a in vec_a))
    len_b = np.sqrt(sum(b**2 for b in vec_b))

    cosine_sim = val_mult / (len_a * len_b)

    return cosine_sim

def calk_similarities(current: str) -> typing.List[typing.Tuple[str, float]]:
    """Вычисляет схожесть книг с заданной"""

    similar = []

    for name, vec in books.items():
        if name == current:
            continue
        cosine_sim = cosine_similarity(books[current], books[name])
        similar.append((name, cosine_sim))
                       
    similar.sort(key=lambda x: x[1], reverse=True)
    return similar


if __name__ == '__main__':
    print(calk_similarities('Грокаем алгоритмы'))
