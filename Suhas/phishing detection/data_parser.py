import numpy as np
from sklearn.model_selection import train_test_split

def load_data():
    # Load the training data from the CSV file
    training_data = np.genfromtxt('dataset.csv', delimiter=',', dtype=np.int32)

    # Extract the inputs from the training data array (all columns but the last one)
    inputs = training_data[:,:-1]

    # Extract the outputs from the training data array (last column)
    outputs = training_data[:, -1]

    # Split the training and testing data
    training_inputs, testing_inputs, training_outputs, testing_outputs = train_test_split(inputs, outputs, test_size=0.2, random_state=21)

    # Return the four arrays
    return training_inputs, training_outputs, testing_inputs, testing_outputs