"""Учебная точка входа для трансформера: низкоуровневые слои отделены от архитектуры."""

# Импортируем низкоуровневые компоненты и функции, чтобы сохранить исходный публичный интерфейс модуля.
from transformer_layers import (
    FeedForwardNetwork,
    LayerNormalization,
    MultiHeadAttention,
    positional_encoding,
    scaled_dot_product_attention,
    softmax,
)
# Импортируем высокоуровневые архитектурные классы из отдельного файла, где теперь собрана общая логика трансформера.
from transformer_model import Decoder, Encoder, Transformer
