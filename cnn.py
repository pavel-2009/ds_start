"""
Реализация кастомной CNN с нуля
"""

print("Загрузка CNN...")

import numpy as np
from matplotlib import pyplot as plt

print("CNN загружена успешно!")

class CNN:

    def __init__(
        self,
        num_classes: int,
        num_filters: int,
        filter_size: int,
        input_shape: tuple,
        learning_rate: float = 0.01
    ):
        self.num_classes = num_classes
        self.num_filters = num_filters

        self.filter_size = filter_size
        self.input_shape = input_shape

        self.learning_rate = learning_rate

        self.filters = []

        self.create_filters()

        self.dense_weights = np.random.rand(num_filters * (input_shape[0] - filter_size + 1) * (input_shape[1] - filter_size + 1), num_classes) * 0.1
        self.dense_bias = np.zeros(num_classes)


    def forward(self, X: np.ndarray, target: np.ndarray = None) -> np.ndarray:
        """Прямой проход через сеть"""

        X = X.reshape(self.input_shape)  # Убедимся, что вход имеет правильную форму
        
        outputs = []
        for filter in self.filters:

            output = self.convolve(X, filter)
            output = self.relu(output)

            outputs.append(output)

        conv_output = np.stack(outputs)  # Собираем выходы всех фильтров в один массив  

        flattened = conv_output.flatten()

        logits = np.dot(flattened, self.dense_weights) + self.dense_bias

        return logits


    def relu(self, X: np.ndarray) -> np.ndarray:
        """ReLU активация"""
        return np.maximum(0, X)
    

    def relu_derivative(self, X: np.ndarray) -> np.ndarray:
        """Производная ReLU для обратного прохода"""
        return (X > 0).astype(float)
    

    def softmax(self, X: np.ndarray) -> np.ndarray:
        """Softmax активация для классификации"""
        exp_X = np.exp(X - np.max(X))  # Стабилизация для предотвращения переполнения
        return exp_X / np.sum(exp_X)
    

    def compute_loss(self, output: np.ndarray, target: np.ndarray) -> float:
        """Вычисление кросс-энтропийной потери"""
        probs = self.softmax(output)
        loss = -np.sum(target * np.log(probs + 1e-15))  # Добавляем маленькое значение для стабильности
        return loss
    

    def compute_dense_input_size(self) -> np.ndarray:
        """Вычисление размера входа для плотных слоев"""
        output_width = self.input_shape[0] - self.filter_size + 1
        output_height = self.input_shape[1] - self.filter_size + 1

        return self.num_filters * output_width * output_height


    def backward(self, d_logits: np.ndarray, conv_output: np.ndarray, X: np.ndarray):
        """Обновление фильтров на основе градиента ошибки"""

        flattened = conv_output.flatten()

        # Вычисляем градиент ПЕРЕД обновлением весов
        d_flattened = np.dot(d_logits, self.dense_weights.T)
        
        # Теперь обновляем Dense слой
        self.dense_weights -= self.learning_rate * np.outer(flattened, d_logits)
        self.dense_bias -= self.learning_rate * d_logits
        d_conv_output = d_flattened.reshape(conv_output.shape)

        d_conv = d_conv_output * self.relu_derivative(conv_output)

        d_out = d_conv.copy()  # Градиент для каждого фильтра

        X = X.reshape(self.input_shape)  # Убедимся, что вход имеет правильную форму

        for filter_idx, filter in enumerate(self.filters):
            d_filter = np.zeros_like(filter)

            for i in range(d_out.shape[1]):
                for j in range(d_out.shape[2]):
                    region = X[i:i+self.filter_size, j:j+self.filter_size]
                    d_filter += d_out[filter_idx, i, j] * region

            self.filters[filter_idx] -= self.learning_rate * d_filter


    def convolve(self, X: np.ndarray, filter: np.ndarray) -> np.ndarray:
        """Применение свертки к входному изображению X с данным фильтром"""
        
        width, height = self.input_shape
        output_width = width - self.filter_size + 1
        output_height = height - self.filter_size + 1

        output = np.zeros((output_width, output_height))

        for i in range(output_width):
            for j in range(output_height):
                region = X[i:i+self.filter_size, j:j+self.filter_size]
                output[i, j] = np.sum(region * filter)  

        return output


    def create_filters(self):
        """Инициализация фильтров случайными значениями"""
        for _ in range(self.num_filters):
            filter = np.random.rand(self.filter_size, self.filter_size) * 0.1

            self.filters.append(filter)

    
    def visualize_feature_maps(self, X: np.ndarray):
        """Визуализация карт признаков после применения фильтров"""
        X = X.reshape(self.input_shape)
        
        fig, axes = plt.subplots(1, self.num_filters, figsize=(4 * self.num_filters, 4))
        if self.num_filters == 1:
            axes = [axes]
        
        for idx, filter in enumerate(self.filters):
            feature_map = self.convolve(X, filter)
            feature_map = self.relu(feature_map)
            
            ax = axes[idx]
            im = ax.imshow(feature_map, cmap='hot')
            ax.set_title(f'Feature Map {idx + 1}')
            ax.axis('off')
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.show()


    def train_step(self, X: np.ndarray, y: np.ndarray):
        """Один шаг обучения на одном примере"""
        X = X.reshape(self.input_shape)
        
        # Forward: свертка
        outputs = []
        for filter in self.filters:
            output = self.convolve(X, filter)
            output = self.relu(output)


            outputs.append(output)
        
        conv_output = np.stack(outputs)
        flattened = conv_output.flatten()
        
        # Forward: Dense слой
        logits = np.dot(flattened, self.dense_weights) + self.dense_bias
        
        # Loss
        loss = self.compute_loss(logits, y)
        
        # Градиент
        d_loss = self.softmax(logits) - y
        
        # Backward
        self.backward(d_loss, conv_output, X)
        
        return loss
    

def train(cnn: CNN, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 10):
    """Простой цикл обучения для CNN"""
    for epoch in range(epochs):
        total_loss = 0
        for X, y in zip(X_train, y_train):
            loss = cnn.train_step(X, y)
            total_loss += loss

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(X_train):.4f}')


if __name__ == "__main__":
    print("Тестирование CNN на случайных данных...")
    cnn = CNN(num_classes=10, num_filters=4, filter_size=3, input_shape=(28, 28))
    X_train = np.random.rand(100, 28, 28)  # 100 изображений 28x28

    cnn.visualize_feature_maps(X_train[0])  # Визуализируем карты признаков для первого изображени

    y_train = np.eye(10)[np.random.randint(0, 10, 100)]  # One-hot labels
    train(cnn, X_train, y_train, epochs=50)
    print("Тестирование завершено.")
