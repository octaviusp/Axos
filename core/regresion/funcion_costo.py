# Funcion de costo de regresion lineal
import numpy as np

def costo(w: np.ndarray, b:np.ndarray, entradas: np.ndarray, esperados: np.ndarray) -> float:
    """
        (MEDIA DEL ERROR CUADRATICO) - MEAN-SQUARED-ERROR
        Descripcion: Esta funcion permite devolver el costo, respecto una serie de parametros,
        y sesgos.

        Parametros:
            w: 1d-vector de los pesos.
            b: 1d-vector de los sesgos.
            entradas: nd-matriz de todas las features de un ejemplar en el entrenamiento (ejemplo si son casas: altura, superficie, etc)
            esperados: 1d-vector con la salida que deberia ir (recordemos que estamos en aprendizaje supervisado).
            Posteriormente la operacion (entradas - esperados) será la diferencia o la distancia de que tan errado estuvo.

        Retorno: numero flotante con la precision del modelo.
        Entre mas cercano a 0 mejor.
    """
    obtenido = np.array([np.dot(w,entrada)+b for entrada in entradas ])

    numero_ejemplos_entrenamiento = entradas.shape[0]

    obtenido = obtenido.reshape(1,3)

    resultado = np.sum(np.square(obtenido - esperados)) / (2 * numero_ejemplos_entrenamiento)
    return resultado