from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


from data_parser import load_data


if __name__ == '__main__':

    # Load the training data
    train_inputs, train_outputs, test_inputs, test_outputs = load_data()

    # Create a decision tree classifier model using scikit-learn
    classifier = tree.DecisionTreeClassifier()

    # Train the decision tree classifier
    classifier.fit(train_inputs, train_outputs)

    # Use the trained classifier to make predictions on the test data
    predictions = classifier.predict(test_inputs)

    # Print the accuracy (percentage of phishing websites correctly predicted)
    accuracy = 100.0 * accuracy_score(test_outputs, predictions)
    print("Accuracy of Decision Tree : " + str(accuracy))
    conf_matrix = confusion_matrix(test_outputs, predictions)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predicted values', fontsize=18)
    plt.ylabel('Actual values', fontsize=18)
    plt.title('Accuracy of Decision Tree : '+str(round(accuracy,2)), fontsize=18)
    plt.show()