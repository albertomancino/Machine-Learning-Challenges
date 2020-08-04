from numpy import array
from numpy import argmax
from keras.utils import to_categorical


# define example
data = [1, 3, 2, 0, 3, 2, 2, 1, 0, 1]

data = array(data)
print('Sample Array of data')
print(data)

# one hot encode
encoded = to_categorical(data)
print('Dati encodati')
print(encoded)


# invert encoding
for i in range( len(encoded) ):
    inverted = argmax(encoded[i])
    print('Dati encodati -> Inversi')
    print(inverted)
