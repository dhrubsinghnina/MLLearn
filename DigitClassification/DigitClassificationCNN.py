import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.datasets import mnist # type: ignore
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load Data
# -------------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape for CNN (samples, height, width, channels)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test  = x_test.reshape(-1, 28, 28, 1)

# -------------------------------
# 2. Define CNN Model
# -------------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------------
# 3. Train Model
# -------------------------------
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2
)

# -------------------------------
# 4. Evaluate Model
# -------------------------------
y_pred = model.predict(x_test)
y_pred_labels = np.argmax(y_pred, axis=1)

# -------------------------------
# 5. Plot Loss & Accuracy
# -------------------------------
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.legend()
plt.show()

# ======================================
# MNIST HANDWRITTEN DIGIT CLASSIFIER
# With Drawing Interface (Tkinter)
# ======================================



import tkinter as tk
from PIL import Image, ImageDraw
# -------------------------------
# 7. DRAWING INTERFACE
# -------------------------------

canvas_size = 280

image = Image.new("L", (canvas_size, canvas_size), 0)  # black background
draw = ImageDraw.Draw(image)

def draw_lines(event):
    x, y = event.x, event.y
    r = 6
    canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
    draw.ellipse([x-r, y-r, x+r, y+r], fill=255)

def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0,0,canvas_size,canvas_size], fill=0)
    result_label.config(text="Draw a digit")

def predict_digit():
    img = image.resize((28,28))
    img = np.array(img) / 255.0
    img = img.reshape(1, 28, 28, 1)

    prediction = model.predict(img)
    digit = np.argmax(prediction)
    confidence = np.max(prediction)

    result_label.config(text=f"Prediction: {digit}   Confidence: {confidence:.2f}")


# -------------------------------
# 8. UI WINDOW
# -------------------------------

root = tk.Tk()
root.title("MNIST Digit Recognizer")

canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg="black")
canvas.pack()

canvas.bind("<B1-Motion>", draw_lines)

btn_frame = tk.Frame(root)
btn_frame.pack()

tk.Button(btn_frame, text="Predict", command=predict_digit, font=("Arial",14)).grid(row=0, column=0, padx=10)
tk.Button(btn_frame, text="Clear", command=clear_canvas, font=("Arial",14)).grid(row=0, column=1, padx=10)

result_label = tk.Label(root, text="Draw a digit", font=("Arial",18))
result_label.pack(pady=10)

root.mainloop()
