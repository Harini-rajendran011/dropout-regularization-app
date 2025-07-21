import streamlit as st
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from utils import create_model, train_and_evaluate, plot_training

st.set_page_config(page_title="Dropout Regularization", layout="wide")
st.title("🧠 Dropout Regularization on MNIST")

# Sidebar configuration
st.sidebar.header("Model Configuration")
dropout_enabled = st.sidebar.checkbox("Use Dropout?", value=True)
dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.9, 0.5, 0.1)
epochs = st.sidebar.slider("Epochs", 1, 10, 3)
batch_size = st.sidebar.slider("Batch Size", 16, 128, 64)

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1,28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
x_train = x_train[:10000].reshape((-1, 28 * 28))
y_train = y_train[:10000]


# Train model and show results
if st.button("🚀 Train Model"):
    with st.spinner("Training in progress... ⏳"):
        model = create_model(dropout_enabled, dropout_rate)
        history, test_acc = train_and_evaluate(model, x_train, y_train, x_test, y_test, epochs, batch_size)

    st.success(f"✅ Test Accuracy: {test_acc * 100:.2f}%")

    # Plot training graph
    st.subheader("📊 Training Graphs")
    fig = plot_training(history)
    st.pyplot(fig)
# Plotting
st.subheader("📈 Training History")
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# Ensure previous plots are cleared
plt.clf()

# Loss plot
ax[0].plot(history.history["loss"], label="Train Loss")
ax[0].plot(history.history["val_loss"], label="Val Loss")
ax[0].set_title("Loss")
ax[0].legend()

# Accuracy plot
ax[1].plot(history.history["accuracy"], label="Train Accuracy")
ax[1].plot(history.history["val_accuracy"], label="Validation Accuracy")
ax[1].set_title("Accuracy")
ax[1].legend()

# Render plot
st.pyplot(fig)
