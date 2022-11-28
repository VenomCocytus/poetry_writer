import numpy as np
import pickle
import tqdm
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
import os


# dataset file path
file_path = "data/sonnets.txt"

# file_path = "data/python_code.py"
basename = "venom"

# load characters dictionaries
char_to_n = pickle.load(open(f"{basename}-char_to_n.pickle", "rb"))
n_to_char = pickle.load(open(f"{basename}-n_to_char.pickle", "rb"))
dict_size = len(char_to_n)

sequence_length = 100

# Build the model
model = Sequential([
    LSTM(700, input_shape=(sequence_length, dict_size), return_sequences=True),
    Dropout(0.2),
    LSTM(700),
    Dropout(0.2),
    Dense(dict_size, activation='softmax'),
])

# load the optimal weights
model.load_weights(f"results/{basename}-{sequence_length}.h5")

# specify the feed to first characters to generate
seed = 'love is war'

n_chars = 500
# Generating characters
generated = ""
for i in tqdm.tqdm(range(n_chars), "Generating text\n"):
    X = np.zeros((1, sequence_length, dict_size))
    for t, char in enumerate(seed):
        X[0, (sequence_length - len(seed)) + t, char_to_n[char]] = 1
    prediction = model.predict(X, verbose=0)[0]
    next_index = np.argmax(prediction)
    next_char = n_to_char[next_index]
    generated += next_char
    seed = seed[1:] + next_char

print("Seed:", seed)
print("Generated text:")
print(generated)