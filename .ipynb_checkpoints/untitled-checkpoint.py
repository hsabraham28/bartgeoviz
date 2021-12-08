import tensorflow as tf

model = tf.keras.models.load_model("occ-log-classification/model")

print(model.predict(['fish tacos']))