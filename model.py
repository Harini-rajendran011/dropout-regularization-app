from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def create_model(use_dropout, dropout_rate):
    model = Sequential()
    model.add(Dense(512, activation="relu", input_shape=(784,)))
    if use_dropout:
        model.add(Dropout(dropout_rate))
    model.add(Dense(256, activation="relu"))
    if use_dropout:
        model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation="softmax"))
    return model
