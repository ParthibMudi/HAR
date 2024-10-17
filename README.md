### Long Short Term Memory (LSTM) Classification of uci_HAR
Test Accuracy = 89.752 in 10 epocs
###  MotionSense Dataset : Smartphone Sensor D
Test Accuracy = 87.898 in 10 epocs

### model v2 - val_accuracy: 0.9075
model = Sequential()	
model.add(GRU(6, input_shape=(WINDOW_LENGTH, NUM_FEATURES), return_sequences=True))	
model.add(GRU(12, return_sequences=True))  	
model.add(GRU(12)) 		
model.add(Dense(NUM_CLASSES, activation='softmax')) 	
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])	
