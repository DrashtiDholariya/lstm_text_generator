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
print("sample text:", text[:500])

# Clean text data
# Keep only lowercase letters and spaces
text = re.sub(r"[^a-z\s]", "",text)

print("after cleaning")
print("sample text:", text[:500])

# Character-level tokenization
# Create sorted list of unique characters
char = sorted(list(set(text)))
print("unique char len:", len(text))

char_to_idx = {c: i for i , c in enumerate(char)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

# Convert text into integer
encoded_text = np.array([char_to_idx[c] for c in text])
print("encoded sample:", encoded_text[:100])

# Create input-output sequences
SEQ_LENGTH = 40
X= []
y= []

# Each input sequence contains SEQ_LENGTH characters
for i in range(len(encoded_text)-SEQ_LENGTH):
    X.append(encoded_text[i:i + SEQ_LENGTH])
    y.append(encoded_text[i + SEQ_LENGTH])
X = np.array(X)
y = to_categorical(y, num_classes = len(char))

print("X shape:", X.shape)
print("Y shape:", y.shape)


# Build the LSTM model
model = Sequential([
    Input(shape=(SEQ_LENGTH,)),
    Embedding(input_dim=len(char), output_dim=50),
    LSTM(128),
    Dense(len(char), activation="softmax")
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
    epochs=5,
    validation_split=0.1
)

#Text generate function
def generate_text(seed, length=500):
    seed = seed.lower()
    result = seed

    for _ in range(length):
        seq = [char_to_idx.get(c, 0) for c in seed]
        seq = pad_sequences([seq], maxlen=SEQ_LENGTH)

        prediction = model.predict(seq, verbose=0)
        next_char = idx_to_char[np.argmax(prediction)]

        result += next_char
        seed = seed[1:] + next_char

    return result

# Generate text from user input
user_seed = input("Enter starting text: ").lower()

if len(user_seed) < SEQ_LENGTH:
    user_seed = user_seed.rjust(SEQ_LENGTH)

output = generate_text(user_seed, 500)
print(output)

with open("sample_output.txt", "w", encoding="utf-8") as f:
  f.write("User Input:\n")
  f.write(user_seed.strip() + "\n\n")
  f.write("Generate Output:\n")
  f.write(output)
  
print("\nSaved to output.txt successfully")