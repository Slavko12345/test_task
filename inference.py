import spacy
import pickle
import torch
from model import LSTM_fixed_len
from shared_utils import encode_sentence


class TargetPredictor:
    def __init__(self):
        #self.tok = spacy.load('en_core_web_sm')
        self.vocab2index = pickle.load(open('models/vocab2index', 'rb'))
        self.vocab_size = pickle.load(open('models/vocab_size', 'rb'))

        self.model = LSTM_fixed_len(self.vocab_size, 50, 50)
        self.model.load_state_dict(torch.load('models/predictor.pt'))
        self.model.eval()

    def predict(self, text):
        encoded = torch.tensor(encode_sentence(text, self.vocab2index)[0])
        return self.model(encoded.unsqueeze(0), 0)[0, 0].item()
