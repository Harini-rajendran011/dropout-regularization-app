import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from model import create_model
from utils import compile_and_train, plot_training


st.set_page_config(page_title="Dropout Regularization Demo", layout="wide")

st.title("🧠 Dropout Regularization Demo on MNIST")

# Sidebar configuration
st.sidebar.header("Model Configuration")
dropout_enabled = st.sidebar.checkbox("Use Dropout?", value=True)
dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.9, 0.5, 0.1)
epochs = st.sidebar.slider("Epochs", 1, 20, 5)
batch_size = st.sidebar.slider("Batch Size", 16, 128, 64)

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape((-1, 28 * 28))
x_test = x_test.reshape((-1, 28 * 28))
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Model building function
def create_model(use_dropout, dropout_rate):
    model = Sequential()
    model.add(Dense(512, activation="relu", input_shape=(784,)))
    if use_dropout:
        model.add(Dropout(dropout_rate))
    model.add(Dense(256, activation="relu"))
    if use_dropout:
        model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

# Train model
if st.button("Train Model"):
    model = create_model(dropout_enabled, dropout_rate)
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        verbose=0
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    st.success(f"🎯 Test Accuracy: {test_acc * 100:.2f}%")

    # Plotting
    st.subheader("📈 Training History")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history.history["loss"], label="Train Loss")
    ax[0].plot(history.history["val_loss"], label="Val Loss")
    ax[0].set_title("Loss")
    ax[0].legend()

    ax[1].plot(history.history["accuracy"], label="Train Acc")
    ax[1].plot(history.history["val_accuracy"], label="Val Acc")
    ax[1].set_title("Accuracy")
    ax[1].legend()

    st.pyplot(fig)
