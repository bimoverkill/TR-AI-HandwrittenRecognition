import tensorflow as tf
import utils
import numpy as np
import matplotlib.pyplot as plt
import random

x_train, y_train = utils.load_data()
x_test, y_test = utils.generate_test_data(x_train, y_train, 1000)

x_test = np.asarray(x_test, dtype=np.float32)
y_test = np.asarray(y_test, dtype=np.float32)

x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.load_model('model/201904122201.mdl')
predictions = model.predict(x_test)

randomnumber = random.randint(0, 999)
print("Prediction : {} | Actually : {}".format(np.argmax(predictions[randomnumber]), y_test[randomnumber]))
plt.imshow(x_test[randomnumber])
plt.show()