import tensorflow as tf
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.datasets import mnist # type: ignore
import numpy as np
import matplotlib.pyplot as plt

# data extraction
(x_train,y_train),(x_test,y_test)=mnist.load_data()
print(f"Shape of X ={x_train.shape}\nShape of y= {y_train.shape}")
print(f"One rows of x[0] :{x_train[0][0]}==Y[0]:{y_train[0]}")
# ploting 1st row:
plt.imshow(x_train[0],cmap='gray')
plt.title(f"Lable:{y_train[0]}")
plt.show()
# Preprocessigng:
# Normalize:
# the pixel values to range of [0,1]
x_train,x_test=x_train/255,x_test/255 
print(f"Normalize x_train={x_train[0]}")
# flatern the images fram 28x28 matrices to a vector:
x_train=x_train.reshape(-1,28*28) # flatten to 784
x_test = x_test.reshape(-1, 28*28)

# defining model
model=Sequential([
    tf.keras.layers.Input(shape=(784,)), # type: ignore
    Dense(units=128,activation='relu'),
    Dense(units=64,activation='relu'),
    Dense(units=10,activation='softmax'),
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print(f"Model summary:\n{model.summary()}")
history=model.fit(x_train,y_train,epochs=100,batch_size=128,validation_split=0.2,verbose=1)
print(f"Keys on history :{history.history.keys()}")
# Making prediction from testing data:
y_pred=model.predict(x_test)

# ploting loss on trainging vs loss on validation
plt.figure()
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# ploting accuracy on training vs acccuracy on validation
plt.figure()
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Convert predictions to labels
y_pred_labels = np.argmax(y_pred, axis=1)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred_labels)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - MNIST")
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
    img = img.reshape(1, 784)

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
