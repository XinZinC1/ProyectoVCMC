import os
import pickle
import numpy 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation,Flatten,Conv2D, MaxPooling2D
from tensorflow.python.keras.callbacks import TensorBoard

#os.environ["CUDA_VISIBLE_DEVICES"]= "-1"
x=pickle.load(open("x.pickle","rb"))
y=pickle.load(open("y.pickle","rb"))

#normalizar pixeles
x=x/255.0
y= numpy.array(y)

neuronas=[32, 64, 128]
densas=[0, 1, 2]
convp=[0, 1, 2]
drop=[0, 1]

model= Sequential()
#neuronas,kernel
model.add(Conv2D(8,(3,3), input_shape=x.shape[1:], activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

model.fit(x,y,batch_size=1, epochs=2, validation_split=0.3)