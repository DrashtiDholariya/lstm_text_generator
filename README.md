# LSTM Text Generation using Shakespeare Dataset

## Overview
This project implements a character-level text generation model using an LSTM neural network. The model is trained on Shakespeare’s complete works and generates new text based on a given seed sequence.

## Dataset
- Source: Project Gutenberg
- Link: https://www.gutenberg.org/ebooks/100
- Format: Plain Text UTF-8

## Approach
- Clean and preprocess raw text
- Perform character-level tokenization
- Create fixed-length input sequences
- Train an LSTM model to predict the next character
- Generate new text iteratively from a seed input

## Model Architecture
- Embedding layer
- LSTM layer (128 units)
- Dense output layer with softmax activation

## How to Run
```bash
pip install -r requirements.txt
python text_generation.py
