import numpy as np
from core.regresion import funcion_costo

def test_costo():
    # Definir datos de prueba
    w = np.array([0.5, 0.8, 1.2])
    b = np.array([0.3])

    entradas = np.array([[5, 6, 7], [4, 5, 6], [7, 8, 9]])
    numero_ejemplos = entradas.shape[0]

    esperados = np.array([5, 12, 21])
    # Calcular el costo
    resultado = funcion_costo.costo(w, b, entradas, esperados)

    predicciones = []

    for i in entradas:
        pesos = 0
        for x in range(3):
            producto = i[x]*w[x]
            pesos += producto
        predicciones.append(pesos+b[0])

    predicciones = np.array(predicciones)
    sumatoria = 0
    for x in range(predicciones.shape[0]):
        sumatoria += (predicciones[x] - esperados[x])**2

    # Verificar el resultado esperado
    sumatoria /= 2*numero_ejemplos
    print("\nFuncion costo: ", resultado)
    print("\nCasero: ", sumatoria)
    assert resultado == sumatoria

# Ejecutar el test
test_costo()