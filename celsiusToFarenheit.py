import tensorflow as tf
import numpy as np

celsius = np.array([-40, -10, 0, 8, 15,22, 38])
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100])

capa = tf.keras.layers.Dense(units = 1, input_shape=[1])
modelo =  tf.keras.Sequential([capa])

modelo.compile(
    optimizer = tf.keras.optimizers.Adam(0.1),
    loss ='mean_squared_error'
)
print("Comenzando entrenamiento")
historial = modelo.fit(celsius,fahrenheit,epochs=1000, verbose = False)
print("Modelo entrenado!!")

from matplotlib import pyplot as plt
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])
plt.show()

print("Hagamos una prediccion!")
resultado = modelo.predict([50.0])
print("El resultado es "+str(resultado) + " fahrenheit!")


