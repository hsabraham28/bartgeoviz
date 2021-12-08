import numpy as np
import pandas as pd
from utils import *
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers.experimental import preprocessing


def really_jank_get_model():
	labeled_ds = csv_to_dataset('occ-log-classification/csv/labeled.csv', batch_size=1, label='Cat', num_epochs=1)
	train_ds = labeled_ds.take(400)

	AUTOTUNE = tf.data.AUTOTUNE
	train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

	vocab_size = 10000
	sequence_length = 20

	vectorize_layer = TextVectorization(
	    standardize="lower_and_strip_punctuation",
	    max_tokens=vocab_size,
	    output_mode='int',
	    output_sequence_length=sequence_length)

	text_ds = train_ds.map(lambda x, y: x['Log'])
	vectorize_layer.adapt(text_ds)

	embedding_dim=5
	hidden_layers=20

	model = Sequential([
	  vectorize_layer,                                         
	  Embedding(vocab_size, embedding_dim, name="embedding"),  
	  GlobalAveragePooling1D(),                                
	  Dense(hidden_layers, activation='relu'),
	  Dense(hidden_layers, activation='relu'),
	  Dense(14, activation="softmax")                          
	])

	model.compile(optimizer='adam',
	            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
	            metrics=['accuracy'])

	model.load_weights('occ-log-classification/checkpoints/my_checkpoint')

	return model



