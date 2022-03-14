import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import csv

# sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# keras
import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import losses


# read dataset
csvfile = open('DSL-StrongPasswordData.csv', 'r')
rdr = csv.reader(csvfile, delimiter=',')
data = []
for row in rdr:
    data.append(row)

# preparing inputs
X = []
for i in range(1, len(data)):
    X.append(data[i][3:])

# preparing outputs
y = []
for i in range(1, len(data)):
    y.append([data[i][0]])

# one hot encoding
ohe = OneHotEncoder()
print(y)
y = ohe.fit_transform(y).toarray()
y = np.asarray(y, dtype=np.uint8)
print(y)

# splitting data -> train 70%, test 15%, validation 15% (total 20400)
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y, test_size=0.15,
                                                    random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size=0.17645,
                                                  random_state=1)

# converting list to ndarray and converting datatypes
X_train = np.asarray(X_train, dtype=np.float)
X_test = np.asarray(X_test, dtype=np.float)
X_val = np.asarray(X_val, dtype=np.float)
y_train = np.asarray(y_train, dtype=np.uint8)
y_test = np.asarray(y_test, dtype=np.uint8)
y_val = np.asarray(y_val, dtype=np.uint8)

# Neural Network Model
model = Sequential()
model.add(Dense(102, input_dim=31, activation='relu'))
model.add(Dense(51, activation='softmax'))

opt = Adam(lr=0.0004)
model.compile(optimizer=opt, loss=losses.categorical_crossentropy,
              metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=20,
                   verbose=0)

history = model.fit(X_train, y_train, epochs=500,
                    validation_data=(X_val, y_val),
                    callbacks=[es], shuffle=False,
                    batch_size=32)

y_pred = model.predict(X_test)


# Converting predictions to label
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))

# Converting one hot encoded test label to label
test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))

# accuracy
a = accuracy_score(pred, test)
print('Accuracy is:', a*100)

# showing the results
print(">>> classification report")
print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

print(">>> confusion matrix")
cf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

cf_matrix = np.asarray(cf_matrix, dtype=np.float)
ax = sn.heatmap(cf_matrix, cmap="Blues")
plt.title("Confusion matrix for 51 classes")
plt.xlabel("Predicted Class")
plt.ylabel("Target Class")
plt.savefig('conf.png')
plt.clf()

# GENERALIZE CONFUSION MATRIX
def generalize(cf_matrix, num_groups):
    group_size = int(len(cf_matrix) / num_groups)
    gen = np.zeros((num_groups, num_groups))
    T = np.zeros(num_groups)

    for i in range(len(cf_matrix)):
        for j in range(len(cf_matrix[0])):
            # find gen_i, gen_j
            for k in range(1, num_groups + 1):
                if (i < len(cf_matrix) / num_groups * k and
                        i >= len(cf_matrix) / num_groups * (k - 1)):
                    gen_i = k - 1
                if (j < len(cf_matrix) / num_groups * k and
                        j >= len(cf_matrix) / num_groups * (k - 1)):
                    gen_j = k - 1
            gen[gen_i][gen_j] += cf_matrix[i][j]
            if i == j:
                T[gen_i] += cf_matrix[i][j]
    
    # distribute false values to neighbor columns
    for i in range(len(gen)):
        for j in range(len(gen[0])):
            if i == j:
                F = gen[i][j] - T[i]
                gen[i][j] -= F
                if j == 0:
                    gen[i][j+1] += F
                elif j == len(gen) - 1:
                    gen[i][j-1] += F
                else:
                    F1 = int(F/2)
                    F2 = F - F1
                    gen[i][j-1] += F1
                    gen[i][j+1] += F2
    return gen

cf_matrix = generalize(cf_matrix, 3)
cf_matrix = np.asarray(cf_matrix, dtype=np.int32)

ax = sn.heatmap(cf_matrix, cmap="Blues", annot=True, fmt='d',
                xticklabels=["1 to 17", "18 to 34", "35 to 51"],
                yticklabels=["1 to 17", "18 to 34", "35 to 51"])
plt.title("Generalized confusion matrix for 51 classes")
plt.xlabel("Predicted Class")
plt.ylabel("Target Class")
plt.savefig('gen_conf.png')
plt.clf()

# normalize and plot confusion matrix
cf_matrix = np.asarray(cf_matrix, dtype=np.float)

for i in range(len(cf_matrix)):
    sum = 0
    for j in range(len(cf_matrix[i])):
        sum += cf_matrix[i][j]
    for j in range(len(cf_matrix[i])):
        cf_matrix[i][j] = cf_matrix[i][j] / sum

ax = sn.heatmap(cf_matrix, cmap="Blues", annot=True,
                xticklabels=["1 to 17", "18 to 34", "35 to 51"],
                yticklabels=["1 to 17", "18 to 34", "35 to 51"],
                vmin=0, vmax=1)
plt.title("Normalized and generalized confusion matrix for 51 classes")
plt.xlabel("Predicted Class")
plt.ylabel("Target Class")
plt.savefig('gen_norm_conf.png')
plt.clf()

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(y_test[0])):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve
plt.figure()
for i in range(len(y_test[0])):
    plt.plot(fpr[i], tpr[i])

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for 51 classes')
plt.savefig('roc_51.png')
plt.clf()

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curve
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Micro-average ROC curve')
plt.legend(loc="lower right")
plt.savefig('micro_avg_roc.png')
plt.clf()

# Plotting loss function
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig('loss.png')
plt.show()
