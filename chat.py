import random
import json
import os

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

intents_path = os.path.join(BASE_DIR, 'intents.json')
model_path = os.path.join(BASE_DIR, 'data.pth')

if not os.path.exists(intents_path):
    raise FileNotFoundError(f"intents.json not found at {intents_path}")

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Trained model not found at {model_path}. "
        "Please run train.py first to generate data.pth."
    )

with open(intents_path, 'r') as json_data:
    intents = json.load(json_data)

data = torch.load(model_path, map_location=device)

input_size  = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words   = data['all_words']
tags        = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"


def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob  = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    return ("I'm not sure I understand your issue. Please describe it in more detail, "
            "or type 'create a ticket' to raise a support request with our IT team.")


if __name__ == "__main__":
    print("IT Support Chatbot - Sam")
    print("=" * 40)
    print("Type 'quit' to exit\n")
    while True:
        sentence = input("You: ")
        if sentence.lower() == "quit":
            break
        resp = get_response(sentence)
        print(f"{bot_name}: {resp}\n")
