import streamlit as st
import torch
import json
import os


# Define the NextWord model class
class NextWord(torch.nn.Module):
    def __init__(self, context_window, vocab_size, emb_dim, hidden_size, activation):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, emb_dim)
        self.lin1 = torch.nn.Linear(context_window * emb_dim, hidden_size)
        self.lin2 = torch.nn.Linear(hidden_size, vocab_size)
        self.activation = activation

    def forward(self, x):
        x = self.emb(x)
        x = x.view(x.shape[0], -1)
        x = torch.relu(self.lin1(x)) if self.activation == "relu" else torch.tanh(self.lin1(x))
        x = self.lin2(x)
        return x


# Load vocabulary from a JSON file
def load_vocab(vocab_path):
    with open(vocab_path, 'r') as vocab_file:
        vocab = json.load(vocab_file)
    return vocab


# Set up the app
st.title("Next Word Prediction Model")
st.write("Generate text using a context-based model with configurable parameters.")

# Load the vocabulary
# Update this line with your specific vocabulary file path
# Model configuration selections
embedding_size = st.selectbox("Embedding Size:", [64, 128])
context_length = st.selectbox("Context Length:", [5, 10, 15])
activation_function = st.selectbox("Activation Function:", ["ReLU", "Tanh"])

# Define fixed model parameters
activation_suffix = f"{activation_function.lower()}"
# Construct the weights path
weights_dir = "Model Weights"  # Update with your weights path
weights_path = f"{weights_dir}/model_weights_c={context_length}_e={embedding_size},{activation_suffix}.pth"
vocab_path = f"Vocabulary/word2idx_c={context_length}_e={embedding_size},{activation_suffix}.json"
vocab = load_vocab(vocab_path)
vocab_size = len(vocab)  # Use the loaded vocabulary size
if activation_function == "Tanh" and embedding_size == 128 and context_length==5:
    hidden_size = 1024
else:
    hidden_size = 128

# Convert the loaded vocabulary to a list or other format if needed
idx2word = {idx: word for word, idx in vocab.items()}


# Load the model with the selected parameters
if os.path.exists(weights_path):
    model = NextWord(context_length, vocab_size, embedding_size, hidden_size, activation_function.lower())
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
else:
    st.error(f"Model weights not found for the selected configuration.")

# User input for text and number of words to predict
input_text = st.text_input("Enter input text:", "Once upon a time")
num_words = st.slider("Number of Next Words to Predict:", 1, 50, 1)


# Define a function for generating multiple next words
def predict_next_words(model, input_text, tokenizer, vocab, num_words):
    input_indices = tokenizer(input_text)
    generated_text = input_text.split()

    for _ in range(num_words):
        # Ensure input context is of the required length
        if len(input_indices) < context_length:
            input_indices = [0] * (context_length - len(input_indices)) + input_indices  # Pad with zeros

        # Convert to tensor and get prediction
        input_tensor = torch.tensor(input_indices[-context_length:]).unsqueeze(0)
        with torch.no_grad():
            logits = model(input_tensor)
            predicted_index = torch.argmax(logits, dim=-1).item()
            predicted_word = idx2word.get(predicted_index, "<unknown>")
            generated_text.append(predicted_word)
            input_indices.append(predicted_index)  # Add new word to context for next prediction

    return " ".join(generated_text[len(input_text.split()):])


# Generate and display the next word predictions
if st.button("Predict Next Words"):
    tokenizer = lambda x: [vocab.get(word, 0) for word in x.split()]  # Simple whitespace tokenizer
    next_words = predict_next_words(model, input_text, tokenizer, vocab, num_words)
    st.write(f"Predicted Next Words: {next_words}")
