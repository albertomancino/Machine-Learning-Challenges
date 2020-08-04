import sys


def InputTargetFeature(csv):            # PRENDE IN INPUT UN OGGETTO CSVMANAGER e restituisce una coppia (nome target, [nome feature1, nome feature2, nome feature3, ...]

    attributi = csv.giveme_attributes()
    print('\nDATASET ATTRIBUTES:')
    print(attributi)

    # ---------- INDICA LA VARIABILE DA PREDIRE ----------
    target = ''
    features = list()
    feature = ''
    polinomials = list()
    polinomial = list()

    # ---------- INSERIMENTO VARIABILE TARGET ----------
    while (attributi.count(target) < 1):
        target = input('Inserisci la variabile TARGET (type \'exit\' to quit)\n')
        if attributi.count(target) < 1 and target != 'exit':
            print('Variabile non presente')
        elif target == 'exit':
            sys.exit()
    print('Target ', target, 'inserito\n')

    # ---------- INSERIMENTO VARIABILI FEATURE ----------
    while feature != 'run' and feature != 'exit' and feature != 'poli':
        feature = ''

        while (attributi.count(feature) < 1 and feature != 'run' and feature != 'exit' and feature != 'poli'):
            feature = input('Inserisci la variabile FEATURE (type \'poli\' to inserti polonomials attributes - type \'run\' to run the algorithm - \'exit\' to quit)\n')

            if attributi.count(feature) < 1 and feature != 'run' and feature != 'exit' and feature != 'poli':
                print('Variabile non presente')

            if features.count(feature) > 0:
                print(features)
                feature = input('Feature già inserita\n')

        if feature == 'exit':  # Inserisci feature se non è la sequenza di uscita 'exit' e se non è già presente
            sys.exit()

        if feature == target:
            print('ATTENZIONE: la feature specificata è uguale alla variabile target.')
            flag = ''
            while (flag != 'Y' and flag != 'N'):
                flag = input('Confermare? (Y/N)\n')
                if flag == 'Y':
                    features.append(feature)
                    print('Feature ', feature , 'inserita')

        elif feature != 'run' and feature != 'exit' and feature != 'poli':
            print('Feature ', feature, 'inserita')
            features.append(feature)

    # ---------- INSERIMENTO VARIABILI POLINOMIALI ----------
    if feature == 'poli':

        print('\nINSERIMENTO VARIABILI POLINOMIALI')

        while feature != 'run' and feature != 'exit':

            feature = ''

            while (attributi.count(feature) < 1 and feature != 'run' and feature != 'exit' and feature != 'next'):
                feature = input(
                    'Inserisci una alla volta le variabili che compongono la polinomiale (type \'run\' to run the algorithm - type \'next to insert another polinomial feature\' - \'exit\' to quit)\n')

                if attributi.count(feature) < 1 and feature != 'run' and feature != 'exit' and feature != 'next':
                    print('Variabile non presente')

            if feature == 'exit':  # Inserisci feature se non è la sequenza di uscita 'exit' e se non è già presente
                sys.exit()

            if feature != 'run' and feature != 'exit' and feature != 'next':
                polinomial.append(feature)

            if feature == 'next' or feature == 'run':
                polinomials.append(polinomial)
                polinomial = list()




    print('Variabile target inserita: ', target)
    print('Feature inserite: ', features)
    print('Feature polinomiali inserite: ', polinomials)

    return [target, features, polinomials]

def normalizationControl():
    
    flag = ''
    
    while flag != 'Y' and flag != 'N':
        flag = input('Normalizzare i dati secondo Z Score Normalization? (Y/N)\n')
        
    return flag