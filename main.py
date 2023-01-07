from sklearn.neural_network import MLPClassifier
import numpy as np

def learn_data():
    inputs = np.genfromtxt('life_expectancy_data.csv', dtype=np.float32, delimiter=',', skip_header=1, usecols=range(3,22))
    outputs = np.genfromtxt('life_expectancy_data.csv', dtype=str, delimiter=',', skip_header=1, usecols=2) == 'Developed'

    # This is the neural network
    classifier = MLPClassifier(random_state=0)
    print()
    test_size = 10

    # Train on all the data AFTER the first 10 (i.e. on 1787 images)
    classifier.fit(inputs[test_size:], outputs[test_size:])

    # Test on ONLY the first 10 digits
    # (which coincidentally are themselves the digits 1,2,3,4,5,6,7,8,9 in order)
    results = classifier.predict(inputs[:test_size])

    print(f'Accuracy: {(results == outputs[:test_size]).mean()}')

if __name__ == '__main__':
    learn_data()
