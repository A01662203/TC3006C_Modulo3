# Importar librerías necesarias para el análisis y visualización de datos
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def generar_datos_con_sesgo(num_samples, bias=True, bias_strength=-2):
    np.random.seed(0)  # Para reproducibilidad de los resultados
    
    # Generar características aleatorias siguiendo una distribución normal estándar
    X = np.random.randn(num_samples, 2)
    
    # Definir los pesos verdaderos (coeficientes) para las características
    true_theta = np.array([1, -3])
    
    # Calcular la combinación lineal de las características y los pesos
    # Esta combinación se ajusta para añadir un término de sesgo (bias_strength)
    # Además, se añade un pequeño ruido aleatorio para hacer los datos más realistas
    linear_combination = X @ true_theta + bias_strength + 0.5 * np.random.randn(num_samples)
    
    # Aplicar la función sigmoide para transformar la combinación lineal en probabilidades
    probabilities = 1 / (1 + np.exp(-linear_combination))
    
    # Convertir las probabilidades en etiquetas binarias (0 o 1) usando un umbral de 0.5
    y = (probabilities >= 0.5).astype(int)
    
    # Si se desea incluir el término de sesgo, agregar una columna de unos a X
    if bias:
        X = np.c_[np.ones((num_samples, 1)), X]
    
    return X, y

# Generar un conjunto de datos de ejemplo con un mayor sesgo (bias_strength positivo)
num_samples = 1000
X, y = generar_datos_con_sesgo(num_samples, bias_strength=2.5)

# Mostrar una gráfica de los datos generados
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 1], X[:, 2], c=y, cmap='viridis')
plt.title('Datos Generados con Mayor Bias')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.show()

# Función sigmoide: convierte una entrada en una probabilidad entre 0 y 1
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Función de costo (log-loss) para la regresión logística
def cost_function(X, y, theta):
    m = len(y)  # Número de muestras
    h = sigmoid(X.dot(theta))  # Predicciones usando la función sigmoide
    epsilon = 1e-5  # Valor pequeño para evitar log(0)
    
    # Calcular el costo usando log-loss, que mide la diferencia entre predicciones y etiquetas verdaderas
    cost = (1/m) * (((-y).T.dot(np.log(h + epsilon))) - ((1-y).T.dot(np.log(1-h + epsilon))))
    
    # Calcular el gradiente, que indica la dirección en la que se debe ajustar theta para minimizar el costo
    grad = (1/m) * (X.T.dot(h-y))
    return cost, grad

# Función de gradiente descendente para optimizar los pesos (coeficientes)
def gradient_descent(X, y, theta, alpha, iters):
    m = len(y)  # Número de muestras
    for i in range(iters):
        h = sigmoid(X.dot(theta))  # Predicciones usando la función sigmoide
        gradient = (1/m) * (X.T.dot(h-y))  # Cálculo del gradiente
        
        # Actualizar los coeficientes en la dirección negativa del gradiente
        theta -= alpha * gradient
        
        # Imprimir el costo cada 1000 iteraciones para monitorear el progreso
        if i % 1000 == 0:
            cost, _ = cost_function(X, y, theta)
            print(f"Cost at iteration {i}", cost)
    return theta

# Función de predicción usando un umbral para determinar la clase (0 o 1)
def predict(X, theta, threshold=0.5):
    return sigmoid(X.dot(theta)) >= threshold

# Inicializar los pesos (coeficientes) en cero
theta = np.zeros(X.shape[1])

# Definir la tasa de aprendizaje (alpha) y el número de iteraciones para el gradiente descendente
alpha = 0.1
iters = 10000

# Calcular el costo inicial con los pesos iniciales
initial_cost, _ = cost_function(X, y, theta)
print(f'Costo inicial: {initial_cost}')

# Ejecutar el algoritmo de gradiente descendente para ajustar los pesos
theta = gradient_descent(X, y, theta, alpha, iters)

# Calcular el costo final después de la optimización
final_cost, _ = cost_function(X, y, theta)
print(f'Costo final: {final_cost}')

# Realizar predicciones usando los pesos optimizados
y_pred = predict(X, theta)

# Calcular la precisión de las predicciones comparando con las etiquetas verdaderas
accuracy = (y_pred == y).mean()
print(f'Precisión: {accuracy}')

# Mostrar los pesos finales después del entrenamiento
print('Pesos finales:')
print(theta)

# Mostrar la frontera de decisión en un gráfico
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 1], X[:, 2], c=y, cmap='viridis')
plt.title('Frontera de Decisión')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')

# Calcular los valores de la frontera de decisión
x_values = [np.min(X[:, 1]), np.max(X[:, 1])]
y_values = - (theta[0] + np.dot(theta[1], x_values)) / theta[2]

# Graficar la frontera de decisión
plt.plot(x_values, y_values, label='Frontera de Decisión')
plt.legend()
plt.show()