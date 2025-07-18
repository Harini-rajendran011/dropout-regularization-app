import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load dataset
@st.cache_data
def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_data()

# Title
st.title("🧠 Dropout Regularization Demo (MNIST)")
st.write("This app compares two neural networks: one with dropout, one without.")

# Dropout rate selection
dropout_rate = st.slider("Select Dropout Rate:", min_value=0.1, max_value=0.7, step=0.1, value=0.5)

# Train button
if st.button("🚀 Train Models"):
    # Model without Dropout
    model_no_dropout = Sequential()
    model_no_dropout.add(Flatten(input_shape=(28, 28)))
    model_no_dropout.add(Dense(512, activation='relu'))
    model_no_dropout.add(Dense(256, activation='relu'))
    model_no_dropout.add(Dense(10, activation='softmax'))

    model_no_dropout.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history_no = model_no_dropout.fit(x_train, y_train, epochs=5, batch_size=128,
                                      validation_data=(x_test, y_test), verbose=0)

    # Model with Dropout
    model_dropout = Sequential()
    model_dropout.add(Flatten(input_shape=(28, 28)))
    model_dropout.add(Dense(512, activation='relu'))
    model_dropout.add(Dropout(dropout_rate))
    model_dropout.add(Dense(256, activation='relu'))
    model_dropout.add(Dropout(dropout_rate))
    model_dropout.add(Dense(10, activation='softmax'))

    model_dropout.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history_do = model_dropout.fit(x_train, y_train, epochs=5, batch_size=128,
                                   validation_data=(x_test, y_test), verbose=0)

    # Plot Accuracy
    st.subheader("📈 Validation Accuracy")
    fig_acc, ax = plt.subplots()
    ax.plot(history_no.history['val_accuracy'], label='No Dropout')
    ax.plot(history_do.history['val_accuracy'], label='With Dropout')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    st.pyplot(fig_acc)

    # Plot Loss
    st.subheader("📉 Validation Loss")
    fig_loss, ax2 = plt.subplots()
    ax2.plot(history_no.history['val_loss'], label='No Dropout')
    ax2.plot(history_do.history['val_loss'], label='With Dropout')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    st.pyplot(fig_loss)

    # Final Evaluation
    _, acc_no = model_no_dropout.evaluate(x_test, y_test, verbose=0)
    _, acc_do = model_dropout.evaluate(x_test, y_test, verbose=0)

    st.success(f"✅ Test Accuracy Without Dropout: {acc_no:.4f}")
    st.success(f"✅ Test Accuracy With Dropout ({dropout_rate}): {acc_do:.4f}")
