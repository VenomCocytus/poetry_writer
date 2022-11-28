# import dependencies
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM

import requests
import os
import pickle


# Create a data folder if it doesn't already exist
if not os.path.exists("data"):
    os.mkdir('data')

# Download 
content = requests.get("https://www.gutenberg.org/cache/epub/1041/pg1041.txt").text # comment if already downloaded

# Save
open("data/sonnets.txt", "w", encoding="utf-8").write(content) # comment if already downloaded

# Data file path
file_path = "data/sonnets.txt"
basename = os.path.basename(file_path)

# Load the data
text = open(file_path, encoding="utf-8").read()
text = text.lower()

# Character Mappings
characters = sorted(list(set(text)))
n_to_char = {n:char for n, char in enumerate(characters)}
char_to_n = {char:n for n, char in enumerate(characters)}

# Print some statistics
n_characters = len(text)
n_unique_characters = len(characters)
print("Unique characters: ", characters)
print("Numbers of unique characters: ", n_unique_characters)
print("Number of characteres", n_characters)

# save these dictionnary for a later use
basename = "venom"
pickle.dump(char_to_n, open(f"{basename}-char_to_n.pickle", "wb"))
pickle.dump(n_to_char, open(f"{basename}-n_to_char.pickle", "wb"))

sequence_length = 100
batch_size = 128
epochs = 100

# Encode the data by converting all text into integers
encoded_text = np.array([char_to_n[c] for c in text])

# Create a custom dataset object
char_ds_object = tf.data.Dataset.from_tensor_slices(encoded_text)

# Printing some data
for characters in char_ds_object.take(6):
    print(characters.numpy(), n_to_char[characters.numpy()])

# Build Sequences by batching
sequences = char_ds_object.batch( 2*sequence_length + 1, drop_remainder=True)

# Print some sequences
for sequence in sequences.take(3):
    print(''.join([n_to_char[i] for i in sequence.numpy()]))

# Sample_splitter function
def sample_splitter(sample):
    length = len(sample)
    ds = tf.data.Dataset.from_tensors((sample[:sequence_length], sample[sequence_length]))
    for i in range(0, length-sequence_length, 1):
        sequence = sample[i: i + sequence_length]
        label = sample[i + sequence_length]
        # extend the dataset with these samples with the concatenate() method
        other_ds = tf.data.Dataset.from_tensors((sequence, label))
        ds = ds.concatenate(other_ds)
    return ds
# Prepare sequences and labels
dataset = sequences.flat_map(sample_splitter)

# One-hot encode the sequences and the labels
def one_hot_encoding(sequence, label):
    return tf.one_hot(sequence, n_unique_characters), tf.one_hot(label, n_unique_characters)

dataset = dataset.map(one_hot_encoding)

# print some samples
for e in dataset.take(3):
    print("Input:", ''.join([n_to_char[np.argmax(char_vector)] for char_vector in e[0].numpy()]))
    print("Target:", n_to_char[np.argmax(e[1].numpy())])
    print("Input shape:", e[0].shape)
    print("Target shape:", e[1].shape)
    print("="*50, "\n")

# repeat, shuffle and batch the dataset
ds = dataset.repeat().shuffle(1024).batch(batch_size, drop_remainder=True)

# building the model
model = Sequential()
model.add(LSTM(700, input_shape=(sequence_length, n_unique_characters), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(700))
model.add(Dropout(0.2))
model.add(Dense(n_unique_characters, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
model.summary()

# define the model path
model_weights_path = f"results/{basename}-{sequence_length}.h5"

# Make a result folder if it doesn't exist or select an existing one
if not os.path.isdir('results'):
    os.mkdir('results')

# Train the model
model.fit(ds, steps_per_epoch=(len(encoded_text) - sequence_length) // batch_size, epochs = epochs)

# Save the model
model.save(model_weights_path)