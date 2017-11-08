from keras.models import Sequential
from keras.layers import Dense
import numpy as np

dataset = np.loadtxt('data.csv', delimiter=',')
# Classify into features and targets
x = dataset[:, 0:8]
y = dataset[:, 8]

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu')
          )               # units in first layer
model.add(Dense(12, activation='sigmoid'))
model.add(Dense(12, activation='relu'))
# units in next layer
model.add(Dense(12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.fit(x, y, epochs=10, batch_size=10)

scores = model.evaluate(x, y)
print('{} and {}'.format(model.metrics_names[1], scores[1]*100))

predictions = model.predict(x)
print(predictions[0])
print(predictions)
