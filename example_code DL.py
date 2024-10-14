# import numpy as np
# from sklearn.datasets import fetch_openml
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from EnergyEfficientAI import EnergyConsumptionML  # Import the class from the file
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.datasets import fetch_openml

# mnist = fetch_openml(data_id=554)

# # Load MNIST data
# # mnist = fetch_openml('mnist_784', as_frame=True)
# X, y = mnist.data.astype('float32').to_numpy(), mnist.target.astype('int')

# # Flatten the images
# X_flatten = np.array([image.flatten() for image in X])

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_flatten, y, test_size=0.2, random_state=42)

# # Define the model (you can pass any model here)
# logreg_model = LogisticRegression()
# cpuIdl = 70
# cpuFull = 170
# # Instantiate the CustomModelTrainer with the model
# model_trainer = EnergyConsumptionML(logreg_model, cpuIdl, cpuFull)

# # Generate the final report by calling generate_report
# model_trainer.generate_report(X_train, y_train, X_test, y_test)
# model_trainer.plot_cpu_usage()

import numpy as np
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
# from EnergyConsumptionDL import EnergyConsumptionDL
from EnergyEfficientAI import EnergyConsumptionDL 
# Load and preprocess the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1) / 255.0  # Normalize pixel values
x_test = np.expand_dims(x_test, axis=-1) / 255.0
y_train = to_categorical(y_train, num_classes=10)  # One-hot encode labels
y_test = to_categorical(y_test, num_classes=10)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Initialize EnergyEfficientDL
energy_tracker = EnergyConsumptionDL(model=model, pcpu_idle=10, pcpu_full=100)

# Generate the report for the training and evaluation process
energy_tracker.generate_report(x_train, y_train, x_test, y_test, epochs=5, batch_size=64)

# Plot CPU usage
energy_tracker.plot_cpu_usage()