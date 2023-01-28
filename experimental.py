from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
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

    best_losses = []
    hidden_layer_sizes = [
        (100,), (100, 100), (100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100, 100), (100,100,100,100), (100,100,100), (100,100), (100)
    ]
    for i, hls in enumerate(hidden_layer_sizes):

        classifier2 = RandomForestClassifier(random_state=0, n_estimators=1)
        classifier2.fit(train_inputs, train_outputs)
        test_results = classifier2.predict(test_inputs)
        train_results = classifier2.predict(train_outputs)


        #classifier = MLPClassifier(random_state=0, hidden_layer_sizes=hls)

        # Train on all the data AFTER the first 10 (i.e. on 1787 images)
        #classifier.fit(train_inputs, train_outputs)

        # Test on ONLY the first 10 digits
        # (which coincidentally are themselves the digits 1,2,3,4,5,6,7,8,9 in order)
        #test_results = classifier.predict(test_inputs)
        #train_results = classifier.predict(train_inputs)

        print(f'\nCurrent iteration: {i}')
        #print(f'Hidden Layer used: {hls}')
        print(f'Test Accuracy: {(test_results == test_outputs).mean() * 100:.5f}')
        print(f'Train Accuracy: {(train_results == train_outputs).mean() * 100:.5f}')
        print(f'Best loss: {classifier2.best_loss_}')
        best_losses.append(classifier2.best_loss_)

        '''
        #plt.title(f'Neural Net Loss with hidden layers: {hls}')
        #plt.plot(range(len(classifier.loss_curve_)), classifier.loss_curve_)
        #plt.xlabel('Iteration Number')
        #plt.ylabel('Loss')
        #plt.show()
        '''

    #plt.title('Best Loss for each number of layers')
    #plt.plot([len(e) for e in hidden_layer_sizes], best_losses)
    #plt.xlabel('Number of layers trained')
    #plt.ylabel('Best Loss Attained')
    #plt.show()


