import numpy as np
import matplotlib.pyplot as plt
import random


def generate_product_data(
        n_products: int = 1000,
        n_features: int = 50,
        density: float = 0.2
    ) -> np.ndarray:
    """
    Генерирует матрицу товаров (n_products × n_features).
    Возвращает numpy array.
    """
    # Случайные числа от 0 до 1
    matrix = np.random.rand(n_products, n_features)
    
    # Маска: обнуляем случайные элементы
    mask = np.random.rand(n_products, n_features) < density
    matrix = matrix * mask
    
    # Округляем до 3 знаков
    matrix = np.round(matrix, 3)
    
    return matrix


def matrix_info(matrix: np.ndarray, name: str = "Матрица"):
    """Выводит информацию о матрице."""
    print(f"\n{name}:")
    print(f"  Форма: {matrix.shape}")
    print(f"  Тип: {matrix.dtype}")
    print(f"  Ненулевых: {np.count_nonzero(matrix)} из {matrix.size} ({np.count_nonzero(matrix)/matrix.size:.1%})")
    print(f"  Мин: {matrix.min():.3f}, Макс: {matrix.max():.3f}, Среднее: {matrix.mean():.3f}")


def compute_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    """Считает матрицу косинусных сходств между всеми товарами."""
    # Нормируем векторы
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Защита от деления на ноль
    norms[norms == 0] = 1
    normalized = matrix / norms
    
    # Матрица сходств = скалярные произведения нормированных векторов
    similarity = normalized @ normalized.T
    
    return similarity


def find_similar_products(
    matrix: np.ndarray,
    target_idx: int,
    top_n: int = 5
) -> list:
    """
    Находит top_n самых похожих товаров для target_idx.
    Возвращает список индексов и сходств.
    """
    # Считаем все попарные сходства
    sim_matrix = compute_similarity_matrix(matrix)
    
    # Берём строку для целевого товара
    similarities = sim_matrix[target_idx]
    
    # Сортируем индексы по убыванию сходства
    similar_indices = np.argsort(similarities)[::-1]
    
    # Исключаем сам товар
    similar_indices = similar_indices[similar_indices != target_idx]
    
    # Берём top_n
    top_indices = similar_indices[:top_n]
    
    return [(idx, similarities[idx]) for idx in top_indices]


def svd_compress(matrix: np.ndarray, k: int):
    """Сжимает матрицу до k признаков с помощью SVD."""

    # SVD разложение
    U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
    
    # Сжимаем: оставляем только k компонент
    U_k = U[:, :k]
    s_k = s[:k]
    
    # Сжатая матрица (товары в новом пространстве)
    compressed = U_k * s_k  # умножаем каждый столбец на соответствующее s
    
    # Доля сохранённой дисперсии
    explained_variance = np.sum(s_k**2) / np.sum(s**2)
    
    return compressed, s_k, explained_variance


def compare_similarities_before_after(
    original: np.ndarray,
    compressed: np.ndarray,
    n_pairs: int = 100
):
    """
    Сравнивает косинусные сходства в исходном и сжатом пространстве.
    """
    n_products = original.shape[0]
    differences = []
    
    for _ in range(n_pairs):
        i, j = random.sample(range(n_products), 2)
        
        # Исходное сходство
        v1_orig = original[i]
        v2_orig = original[j]
        sim_orig = np.dot(v1_orig, v2_orig) / (np.linalg.norm(v1_orig) * np.linalg.norm(v2_orig))
        
        # Сходство в сжатом пространстве
        v1_comp = compressed[i]
        v2_comp = compressed[j]
        sim_comp = np.dot(v1_comp, v2_comp) / (np.linalg.norm(v1_comp) * np.linalg.norm(v2_comp))
        
        differences.append(abs(sim_orig - sim_comp))
    
    return {
        'mean_diff': np.mean(differences),
        'std_diff': np.std(differences),
        'max_diff': np.max(differences)
    }


def interpret_topics(U: np.ndarray, s: np.ndarray, Vt: np.ndarray, 
                     feature_names: list = None, top_n: int = 5):
    """Анализирует, что означают скрытые темы."""

    if feature_names is None:
        feature_names = [f"признак_{i}" for i in range(Vt.shape[1])]
    
    print(f"\nСкрытые темы (первые {min(3, len(s))} из {len(s)}):")
    for topic_idx in range(min(3, len(s))):
        topic = Vt[topic_idx]
        
        # Находим признаки с наибольшим весом в этой теме
        important = sorted(
            [(feature_names[i], abs(topic[i])) for i in range(len(topic))],
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        print(f"\nТема {topic_idx + 1} (важность = {s[topic_idx]:.2f}):")
        for feat, weight in important:
            print(f"  {feat}: {weight:.3f}")



def plot_products_2d(compressed: np.ndarray, title: str = "Товары в 2D"):
    """Рисует товары на плоскости (первые две компоненты)."""
    plt.figure(figsize=(10, 8))
    plt.scatter(compressed[:, 0], compressed[:, 1], alpha=0.5, s=30)
    plt.title(title)
    plt.xlabel("Первая компонента")
    plt.ylabel("Вторая компонента")
    plt.grid(True)
    plt.show()


def plot_singular_values(s: np.ndarray, explained_variance: float = None):
    """Рисует сингулярные числа и накопленную дисперсию."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Сингулярные числа
    axes[0].plot(s, 'bo-')
    axes[0].set_title("Сингулярные числа")
    axes[0].set_xlabel("Номер")
    axes[0].set_ylabel("Значение")
    axes[0].grid(True)
    
    # Накопленная дисперсия
    cumsum = np.cumsum(s**2) / np.sum(s**2)
    axes[1].plot(cumsum, 'ro-')
    axes[1].axhline(y=0.9, color='gray', linestyle='--', label='90%')
    axes[1].set_title("Накопленная дисперсия")
    axes[1].set_xlabel("Количество компонент")
    axes[1].set_ylabel("Доля информации")
    axes[1].grid(True)
    axes[1].legend()
    
    if explained_variance:
        axes[1].axhline(y=explained_variance, color='blue', linestyle=':', 
                       label=f'Текущее сжатие: {explained_variance:.1%}')
        axes[1].legend()
    
    plt.tight_layout()
    plt.show()



def main():
    """Запускает все тесты."""
    print("="*70)
    print("ДЕНЬ 2. МАТРИЦЫ И SVD (ВСЁ ЧЕРЕЗ NUMPY)")
    print("="*70)
    
    # 1. Генерируем данные
    print("\n[1] ГЕНЕРАЦИЯ ДАННЫХ")
    matrix = generate_product_data(n_products=500, n_features=30, density=0.2)
    matrix_info(matrix, "Матрица товаров")
    
    # 2. Проверяем косинусные сходства
    print("\n[2] ПОИСК ПОХОЖИХ ТОВАРОВ")
    target = 0
    similar = find_similar_products(matrix, target, top_n=5)
    print(f"Товар {target} похож на:")
    for idx, sim in similar:
        print(f"  Товар {idx}: сходство {sim:.4f}")
    
    # 3. SVD и сжатие
    print("\n[3] SVD И СЖАТИЕ")
    
    # Пробуем разные k
    for k in [3, 5, 10, 15]:
        compressed, s_k, expl_var = svd_compress(matrix, k)
        print(f"\nk = {k}: сохранено {expl_var:.1%} информации")
        
        # Сравниваем сходства
        stats = compare_similarities_before_after(matrix, compressed, n_pairs=200)
        print(f"  Ошибка сходства: средняя {stats['mean_diff']:.4f} ± {stats['std_diff']:.4f}")
    
    # 4. Интерпретация тем (на примере k=5)
    print("\n[4] ИНТЕРПРЕТАЦИЯ ТЕМ")
    U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
    
    # Создаём осмысленные названия признаков
    feature_names = [f"жанр_{i}" for i in range(matrix.shape[1])]
    interpret_topics(U[:, :5], s[:5], Vt[:5, :], feature_names, top_n=4)
    
    # 5. Визуализация
    print("\n[5] ВИЗУАЛИЗАЦИЯ")
    
    # Сжимаем до 2D для визуализации
    compressed_2d, s_2d, _ = svd_compress(matrix, 2)
    plot_products_2d(compressed_2d, "500 товаров в 2D (первые две компоненты SVD)")
    
    # График сингулярных чисел
    plot_singular_values(s)



if __name__ == "__main__":
    main()