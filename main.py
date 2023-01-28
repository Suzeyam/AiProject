from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt

def learn_data():
    inputs = np.genfromtxt(
        'life_expectancy_data.csv', dtype=np.float32, delimiter=',', skip_header=1, usecols=range(3, 22)
    )
    outputs = np.genfromtxt(
        'life_expectancy_data.csv', dtype=str, delimiter=',', skip_header=1, usecols=2
    ) == 'Developed'

    test_size = 240
    inputs, outputs = shuffle(inputs, outputs, random_state=10)

    train_inputs = inputs[test_size:]
    train_outputs = outputs[test_size:]
    test_inputs = inputs[:test_size]
    test_outputs = outputs[:test_size]

    best_train_losses = []
    best_validation_losses = []
    hidden_layer_sizes = [(100,) * i for i in range(1, 20)]
    for i, hls in enumerate(hidden_layer_sizes):

        classifier = MLPClassifier(random_state=0, hidden_layer_sizes=hls)

        # Train on all the data AFTER the first 10 (i.e. on 1787 images)
        classifier.fit(train_inputs, train_outputs)

        # Test on ONLY the first 10 digits
        # (which coincidentally are themselves the digits 1,2,3,4,5,6,7,8,9 in order)
        test_results = classifier.predict(test_inputs)
        train_results = classifier.predict(train_inputs)

        print(f'\nCurrent iteration: {i}')
        print(f'Hidden Layer used: {hls}')
        print(f'Test Accuracy: {(test_results == test_outputs).mean() * 100:.5f}')
        print(f'Train Accuracy: {(train_results == train_outputs).mean() * 100:.5f}')
        print(f'Best loss: {classifier.best_loss_}')
        best_train_losses.append(classifier.best_loss_)
        best_validation_losses.append(classifier.validation_scores_.max())

        '''
        plt.title(f'Neural Net Loss with hidden layers: {hls}')
        plt.plot(range(len(classifier.loss_curve_)), classifier.loss_curve_)
        plt.xlabel('Iteration Number')
        plt.ylabel('Loss')
        plt.show()
        '''

    plt.title('Best Losses for each number of layers')
    plt.plot(
        [len(e) for e in hidden_layer_sizes],
        best_train_losses,
        label='Train loss'
    )
    plt.plot(
        [len(e) for e in hidden_layer_sizes],
        best_validation_losses,
        label='Validation loss'
    )
    plt.xlabel('Number of layers trained')
    plt.ylabel('Best Loss Attained')
    plt.show()

if __name__ == '__main__':
    learn_data()
