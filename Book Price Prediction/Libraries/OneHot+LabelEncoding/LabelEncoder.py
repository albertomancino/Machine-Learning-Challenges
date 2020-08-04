from sklearn import preprocessing

# preparo label encoding
le = preprocessing.LabelEncoder()

#dati da encodare
data = ["paris", "paris", "tokyo", "amsterdam","Amsterdam","amsterdam"]

le.fit(data)

print("Classi univoche individuate nel dataset")
print( le.classes_ ) #print delle classi univoche individuate_

transofrmed_data = le.transform(data)

print('Dataset originale')
print(data)
print('Dataset trasformato')
print(transofrmed_data)

print('Effettuo la trasformazione inversa')
transofrmed_data_inverted = le.inverse_transform(transofrmed_data)
print(transofrmed_data_inverted)



