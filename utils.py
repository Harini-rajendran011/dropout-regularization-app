from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

def train_and_evaluate(model, x_train, y_train, x_test, y_test, epochs, batch_size):
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy']
    )

    history = model.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    return history, test_acc

def plot_training(history):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(history.history["loss"], label="Train Loss")
    ax[0].plot(history.history["val_loss"], label="Val Loss")
    ax[0].set_title("Loss")
    ax[0].legend()

    ax[1].plot(history.history["accuracy"], label="Train Accuracy")
    ax[1].plot(history.history["val_accuracy"], label="Validation Accuracy")
    ax[1].set_title("Accuracy")
    ax[1].legend()

    return fig
