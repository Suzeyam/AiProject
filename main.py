from sklearn.neural_network import MLPClassifier
import numpy as np

def learn_data():
    inputs = np.genfromtxt('life_expectancy_data.csv', dtype=np.float32, delimiter=',', skip_header=1, usecols=range(3,22))
    outputs = np.genfromtxt('life_expectancy_data.csv', dtype=str, delimiter=',', skip_header=1, usecols=2) == 'Developed'

    # This is the neural network
    classifier = MLPClassifier(random_state=0)
    test_size = 240

    train_inputs = inputs[test_size:]
    train_outputs = outputs[test_size:]
    test_inputs = inputs[:test_size]
    test_outputs = outputs[:test_size]

    # Train on all the data AFTER the first 10 (i.e. on 1787 images)
    classifier.fit(train_inputs, train_outputs)

    # Test on ONLY the first 10 digits
    # (which coincidentally are themselves the digits 1,2,3,4,5,6,7,8,9 in order)
    results = classifier.predict(test_inputs)

    print(f'Accuracy: {(results == test_outputs).mean()}')

if __name__ == '__main__':
    learn_data()
