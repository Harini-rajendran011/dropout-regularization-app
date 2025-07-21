import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

def create_model(use_dropout, dropout_rate):
    model = Sequential()
    model.add(Dense(512, activation="relu", input_shape=(784,)))
    if use_dropout:
        model.add(Dropout(dropout_rate))
    model.add(Dense(256, activation="relu"))
    if use_dropout:
        model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])
    return model

def train_and_evaluate(model, x_train, y_train, x_test, y_test, epochs, batch_size):
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=0)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    return history, test_acc

def plot_training(history):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    ax[0].plot(history.history["loss"], label="Train Loss")
    ax[0].plot(history.history["val_loss"], label="Val Loss")
    ax[0].set_title("Loss")
    ax[0].legend()

    ax[1].plot(history.history["accuracy"], label="Train Acc")
    ax[1].plot(history.history["val_accuracy"], label="Val Acc")
    ax[1].set_title("Accuracy")
    ax[1].legend()

    return fig
