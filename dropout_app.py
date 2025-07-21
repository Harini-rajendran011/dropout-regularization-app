import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical

# Page config
st.set_page_config(page_title="Dropout Regularization Demo", layout="centered")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .stSlider > div {
        color: #ff4b4b;
    }
    </style>
    """, unsafe_allow_html=True
)

# App Title
st.markdown("## 🧠 Dropout Regularization Demo (MNIST)")
st.write("This app compares two neural networks: one with dropout, one without.")

# Dropout slider
dropout_rate = st.slider("Select Dropout Rate:", 0.1, 0.7, 0.5, 0.05)

# Load Data
@st.cache_data
def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train.reshape(-1, 28 * 28)
    x_test = x_test.reshape(-1, 28 * 28)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_data()

# Build Models
def build_model(dropout=False):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    if dropout:
        model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train Models on Button Click
if st.button("🚀 Train Models"):
    with st.spinner("Training in progress... please wait ⏳"):
        model_no_dropout = build_model(dropout=False)
        history_no_dropout = model_no_dropout.fit(
            x_train, y_train, epochs=10, batch_size=128,
            validation_data=(x_test, y_test), verbose=0
        )

        model_dropout = build_model(dropout=True)
        history_dropout = model_dropout.fit(
            x_train, y_train, epochs=10, batch_size=128,
            validation_data=(x_test, y_test), verbose=0
        )

    st.success("Training complete ✅")

    # Plot Results
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy
    axs[0].plot(history_no_dropout.history['val_accuracy'], label="Without Dropout")
    axs[0].plot(history_dropout.history['val_accuracy'], label="With Dropout")
    axs[0].set_title("Validation Accuracy")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend()

    # Loss
    axs[1].plot(history_no_dropout.history['val_loss'], label="Without Dropout")
    axs[1].plot(history_dropout.history['val_loss'], label="With Dropout")
    axs[1].set_title("Validation Loss")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Loss")
    axs[1].legend()

    st.pyplot(fig)
