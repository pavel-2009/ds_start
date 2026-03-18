"""Подробно прокомментированные высокоуровневые блоки трансформера."""

# Импортируем NumPy для типизации массивов и единообразия интерфейса с низкоуровневыми компонентами.
import numpy as np

# Импортируем базовые строительные блоки из отдельного модуля, чтобы архитектурная композиция была отделена от математических примитивов.
from transformer_layers import FeedForwardNetwork, LayerNormalization, MultiHeadAttention


# Объявляем класс кодировщика, который последовательно применяет self-attention, residual-связи, нормализацию и feed-forward блоки.
class Encoder:
    # Определяем конструктор, чтобы задать глубину кодировщика и размеры его внутренних представлений.
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int):
        # Сохраняем число слоев кодировщика для цикла в прямом проходе.
        self.num_layers = num_layers
        # Сохраняем размерность модели как общий размер скрытых представлений.
        self.d_model = d_model
        # Создаем список слоев многоголового внимания: по одному на каждый слой кодировщика.
        self.attention_layers = [MultiHeadAttention(d_model, num_heads) for _ in range(num_layers)]
        # Создаем список feed-forward блоков: каждый слой кодировщика имеет собственный нелинейный преобразователь.
        self.ffn_layers = [FeedForwardNetwork(d_model, d_ff) for _ in range(num_layers)]
        # Создаем список нормализаций, сохраняя исходную упрощенную структуру с одним списком нормализационных модулей.
        self.norm_layers = [LayerNormalization(d_model) for _ in range(num_layers)]

    # Определяем прямой проход кодировщика.
    def forward(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        # Проходим по всем слоям кодировщика сверху вниз, последовательно уточняя представление последовательности.
        for i in range(self.num_layers):
            # Сначала вычисляем self-attention, где запросы, ключи и значения совпадают и берутся из текущего состояния x.
            attention_output, _ = self.attention_layers[i].forward(x, x, x, mask)
            # Добавляем residual-связь и нормализуем результат, чтобы стабилизировать обучение и сохранить исходную информацию.
            x = self.norm_layers[i].forward(x + attention_output)
            # Затем пропускаем каждую позицию через feed-forward сеть для дополнительного нелинейного преобразования.
            ffn_output = self.ffn_layers[i].forward(x)
            # Снова добавляем residual-связь и применяем нормализацию, сохраняя общую схему исходной реализации.
            x = self.norm_layers[i].forward(x + ffn_output)
        # Возвращаем выход кодировщика, который далее может быть передан декодировщику.
        return x


# Объявляем класс декодировщика, который содержит masked self-attention, cross-attention и feed-forward блоки.
class Decoder:
    # Определяем конструктор, чтобы настроить количество слоев и размерности внутренних компонентов.
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int):
        # Сохраняем число слоев декодировщика.
        self.num_layers = num_layers
        # Сохраняем размерность модели для согласованности с кодировщиком и входными данными.
        self.d_model = d_model
        # Создаем первый набор attention-слоев для masked self-attention внутри декодировщика.
        self.attention_layers1 = [MultiHeadAttention(d_model, num_heads) for _ in range(num_layers)]
        # Создаем второй набор attention-слоев для внимания к выходу кодировщика.
        self.attention_layers2 = [MultiHeadAttention(d_model, num_heads) for _ in range(num_layers)]
        # Создаем feed-forward блоки по одному на каждый слой декодировщика.
        self.ffn_layers = [FeedForwardNetwork(d_model, d_ff) for _ in range(num_layers)]
        # Создаем нормализационные слои, сохраняя исходную структуру реализации.
        self.norm_layers = [LayerNormalization(d_model) for _ in range(num_layers)]

    # Определяем прямой проход декодировщика.
    def forward(self, x: np.ndarray, enc_output: np.ndarray, mask1: np.ndarray = None, mask2: np.ndarray = None) -> np.ndarray:
        # Последовательно обрабатываем вход через все слои декодировщика.
        for i in range(self.num_layers):
            # Сначала выполняем masked self-attention, чтобы декодер видел только разрешенные токены собственной последовательности.
            attention_output1, _ = self.attention_layers1[i].forward(x, x, x, mask1)
            # Добавляем residual-связь и нормализуем результат первого attention-блока.
            x = self.norm_layers[i].forward(x + attention_output1)
            # Затем выполняем cross-attention, где запросы берутся из декодера, а ключи и значения — из кодировщика.
            attention_output2, _ = self.attention_layers2[i].forward(x, enc_output, enc_output, mask2)
            # Снова применяем residual-связь и нормализацию после внимания к кодировщику.
            x = self.norm_layers[i].forward(x + attention_output2)
            # Пропускаем результат через feed-forward сеть для позиционно-независимого нелинейного преобразования.
            ffn_output = self.ffn_layers[i].forward(x)
            # Завершаем слой еще одной residual-связью и нормализацией.
            x = self.norm_layers[i].forward(x + ffn_output)
        # Возвращаем итоговое состояние декодировщика.
        return x


# Объявляем объединяющий класс Transformer, который связывает кодировщик и декодировщик в одну архитектуру.
class Transformer:
    # Определяем конструктор, чтобы собрать внутренние блоки модели из заданных гиперпараметров.
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int):
        # Создаем экземпляр кодировщика с указанной глубиной и размерностями.
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff)
        # Создаем экземпляр декодировщика с теми же размерностями для совместимости представлений.
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff)

    # Определяем полный прямой проход через трансформер.
    def forward(self, enc_input: np.ndarray, dec_input: np.ndarray, enc_mask: np.ndarray = None, dec_mask: np.ndarray = None) -> np.ndarray:
        # Сначала пропускаем вход кодировщика через encoder, чтобы получить контекстное представление исходной последовательности.
        enc_output = self.encoder.forward(enc_input, enc_mask)
        # Затем пропускаем вход декодировщика и контекст кодировщика через decoder, чтобы получить итоговое представление выходной последовательности.
        dec_output = self.decoder.forward(dec_input, enc_output, dec_mask)
        # Возвращаем результат декодировщика как выход всей модели.
        return dec_output
