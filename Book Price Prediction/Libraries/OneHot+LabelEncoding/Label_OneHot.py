from numpy import argmax
from keras.utils import to_categorical
from sklearn import preprocessing

# preparo label encoding
le = preprocessing.LabelEncoder()

# feature
data = ["paris", "paris", "tokyo", "amsterdam","Amsterdam","AMSTERDAM"]
print(data)

# lowercase della feature
data = [x.lower() for x in data]

# preparo il label encoding
le.fit(data)

print("Classi univoche individuate nel dataset")
print( le.classes_ ) #print delle classi univoche individuate_

transofrmed_data = le.transform(data)

print('Dataset originale')
print(data)
print('Dataset trasformato')
print(transofrmed_data)


# one hot encode
encoded = to_categorical(transofrmed_data)
print('Dati encodati')
print(encoded)

'''EFFETTUO QUI I MIEI ALGORITMI DI ML SUL DATASET APPENA PROCESSATO'''

# invert One Hot Encoding
for i in range( len(encoded) ):
    inverted = argmax(encoded[i])
    print('Dati invers-encoded ', inverted)

# inverse Label Encoding
print('Effettuo la trasformazione inversa')
transofrmed_data_inverted = le.inverse_transform(transofrmed_data)
print(transofrmed_data_inverted)
