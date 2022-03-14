import models
import pickle
import matplotlib.pyplot as plt

file = open('data/flowdata.pickle', 'rb')
sd = pickle.load(file)
X, Y, XT, YT = sd[0], sd[1], sd[2], sd[3]


model = models.DTModel(X, Y, XT, YT)
decision_acc = model.run()

model = models.NBModel(X, Y, XT, YT)
bayes_acc = model.run()

model = models.SVMModel(X, Y, XT, YT)
svm_acc = model.run()

model = models.KNNModel(X, Y, XT, YT)
knn_acc = model.run()

model = models.LogModel(X, Y, XT, YT)
log_acc = model.run()

model = models.ANNModel(X, Y, XT, YT)
ann_acc = model.run()

data = {'Decision Tree': decision_acc, 'Gaussian Naive Bayes': bayes_acc,
        'SVM': svm_acc, 'KNN': knn_acc, 'Logistic Regression': log_acc, 'ANN': ann_acc}


data_values = list(data.values())
c = ['#3395FF', '#33FFCC', '#87FF33', '#FFE333', '#FF9F33', '#FF6133']

plt.bar(data.keys(), data_values, color=c)
for v, i in enumerate(data_values):
    plt.text(v, i+2, " "+str(data_values[v]),
             color='black', va='center', fontweight='bold')

plt.xlabel("Results")
plt.ylabel("Accuracy")
plt.xticks(rotation=20, ha='right')
plt.title("Model Type")
plt.autoscale()
# plt.savefig('results.png')
plt.show()
