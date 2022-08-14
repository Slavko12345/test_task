import argparse
import pickle
import torch

from train_utils import get_train_val_dataloaders, train_model
from model import LSTM_fixed_len

if __name__ == '__main__':
    """
    Trains LSTM model for prediction of text readability.
    """
    parser = argparse.ArgumentParser(description='Training LSTM model')
    parser.add_argument("-n", "--num_epochs", type=int, default=20,
                        help="Number of epochs to train")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01,
                        help="Learning rate")
    args = parser.parse_args()

    # creating train and validation dataloaders
    train_dl, test_dl, vocab2index, vocab_size = get_train_val_dataloaders()

    # initiating LSTM model
    model = LSTM_fixed_len(vocab_size, 50, 50)

    # training model for a fixed number of epochs
    train_model(model=model,
                train_dl=train_dl,
                test_dl=test_dl,
                epochs=args.num_epochs,
                lr=args.learning_rate)

    # saving auxiliary files needed for inference
    pickle.dump(vocab2index, open('models/vocab2index', 'wb'))
    pickle.dump(vocab_size, open('models/vocab_size', 'wb'))
    # saving model
    torch.save(model.state_dict(), 'models/predictor.pt')
