from sklearn import datasets
from keras.models import Sequential
from keras.layers import Dense

iris = datasets.load_iris()

x = iris.data
y = iris.target

model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
			  optimizer='adam', metrics=['accuracy'])
model.fit(x, y, batch_size=15, epochs=150)
scores = model.evaluate(x, y)

predictions = model.predict(x)

rounded = [round(x[0]) for x in predictions]
count = 0
for i in range(len(predictions)):
	if(y[i] != predictions[i]):
		count = count + 1
print('Number of wrong answers : {}'.format(count))
