### Long Short Term Memory (LSTM) Classification of uci_HAR
Test Accuracy = 89.752 in 10 epocs
###  MotionSense Dataset : Smartphone Sensor D
Test Accuracy = 87.898 in 10 epocs

### model v2 - val_accuracy: 0.9075
model = Sequential()
model.add(GRU(6, input_shape=(WINDOW_LENGTH, NUM_FEATURES), return_sequences=True))

# Add more layers as needed, e.g., additional GRU layers or dense layers
model.add(GRU(12, return_sequences=True))  # Another GRU layer
model.add(GRU(12))  # Final GRU layer without returning sequences
model.add(Dense(NUM_CLASSES, activation='softmax'))  # Output layer for classification

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
