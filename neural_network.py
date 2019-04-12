import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import utils
import datetime
from random import randint

time = datetime.datetime.now()

x_train, y_train = utils.load_data()
x_test, y_test = utils.generate_test_data(x_train, y_train, 1000)

x_train = np.asarray(x_train, dtype=np.float32)
y_train = np.asarray(y_train, dtype=np.float32)
x_test = np.asarray(x_test, dtype=np.float32)
y_test = np.asarray(y_test, dtype=np.float32)


x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)

loss, acc = model.evaluate(x_test, y_test)
print("loss : {}\nacc : {}".format(loss, acc))
model.save("model/{}.mdl".format(time.strftime("%Y%m%d%H%M")))

predictions = model.predict(x_test)

randomnumber = randint(0, 999)
print("Prediction : {} | Actually : {}".format(np.argmax(predictions[randomnumber]), y_test[randomnumber]))
plt.imshow(x_test[randomnumber])
plt.show()