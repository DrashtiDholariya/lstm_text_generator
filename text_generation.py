import re
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input
from tensorflow.keras.utils import to_categorical

# Load shakespeare text data set
with open("data/shakespeare.txt", "r", encoding="utf-8") as file:
    text = file.read().lower()
    
print("total char:", len(text))
print("sample text:", text[:300])

# Clean text data
# Keep only lowercase letters and spaces
text = re.sub(r"[^a-z\s]", "", text)

# Limit data so CPU can train properly
text = text[:300000]

print("after cleaning")
print("sample text:", text[:300])

# Character-level tokenization
# Create sorted list of unique characters
chars = sorted(list(set(text)))
print("unique char len:", len(chars))

char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

# Convert text into integer format
encoded_text = np.array([char_to_idx[c] for c in text])
print("encoded sample:", encoded_text[:100])

# Create input-output sequences
SEQ_LENGTH = 40
X = []
y = []

# Each input sequence contains SEQ_LENGTH characters
for i in range(len(encoded_text) - SEQ_LENGTH):
    X.append(encoded_text[i:i + SEQ_LENGTH])
    y.append(encoded_text[i + SEQ_LENGTH])

X = np.array(X)
y = to_categorical(y, num_classes=len(chars))

print("X shape:", X.shape)
print("Y shape:", y.shape)

# Build the LSTM model
model = Sequential([
    Input(shape=(SEQ_LENGTH,)),
    Embedding(input_dim=len(chars), output_dim=50),
    LSTM(128),
    Dense(len(chars), activation="softmax")
])

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam"
)

model.summary()

# Train model
model.fit(
    X, y,
    batch_size=128,
    epochs=20,
    validation_split=0.1
)

# Function for temperature-based sampling
# This helps avoid repeated words
def sample_with_temperature(preds, temperature=0.8):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)

# Text generation function
def generate_text(seed, length=300, temperature=0.8):
    seed = seed.lower()
    result = seed

    for _ in range(length):
        seq = [char_to_idx.get(c, 0) for c in seed]
        seq = pad_sequences([seq], maxlen=SEQ_LENGTH)

        prediction = model.predict(seq, verbose=0)[0]
        next_index = sample_with_temperature(prediction, temperature)
        next_char = idx_to_char[next_index]

        result += next_char
        seed = seed[1:] + next_char

    return result

# Generate text from user input
user_seed = input("Enter starting text: ").lower()

if len(user_seed) < SEQ_LENGTH:
    user_seed = user_seed.rjust(SEQ_LENGTH)

gen_length = int(input("Enter number of characters to generate: "))

output = generate_text(user_seed, gen_length, temperature=0.8)
print("\nGenerated Text:\n")
print(output)

with open("sample_output.txt", "w", encoding="utf-8") as f:
    f.write("User Input:\n")
    f.write(user_seed.strip() + "\n\n")
    f.write("Generated Output:\n")
    f.write(output)

print("\nSaved to sample_output.txt successfully")
