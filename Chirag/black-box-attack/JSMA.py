from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools

from cleverhans.utils_tf import model_train, model_eval
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks_tf import jacobian_graph
from cleverhans.attacks import SaliencyMapMethod

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.python.platform import flags
from tensorflow.keras.models import load_model

from tensorflow import keras

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report


def prepare_data(word2vec_seq_path):
    """Doing initial data preprocessing"""
    word2vec_df = pd.read_csv(word2vec_seq_path)
    headers = list(word2vec_df)
    headers.remove('Result_False')
    headers.remove('Result_True')

    X = np.array(word2vec_df[headers].values.tolist())
    y = np.array(word2vec_df[['Result_False', 'Result_True']].values.tolist())

    # Split data to training (70%) and testing (30%)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    print("Value counts for training \n")
    print(y_train[:, 0].size)
    print("\n")
    print("Value counts for testing \n")
    print(y_test[:, 0].size)

    return headers, X_train, X_test, y_train, y_test


def classify(X_train, y_train, X_test, y_test):
    """Generates outputs for input data"""

    # Neural Net
    model = load_model('weights/spamModel.h5', compile=False)

    prediction = model.predict(X_test)

    res = []
    for i in prediction:
        if round(i[0]) == 1:
            res.append([0, 1])
        else:
            res.append([1, 0])
    prediction = np.array(res)

    return prediction


def mlp_model(input_shape, input_ph=None, logits=False):
    """MultiLayer  Perceptron  model to generate perturbations"""
    model = Sequential()

    layers = [
        Dense(128, activation='relu', input_shape=input_shape),  # 256
        Dropout(0.4),
        Dense(128,  activation='relu'),  # 256
        Dropout(0.4),
        Dense(FLAGS.nb_classes),
    ]

    for l in layers:
        model.add(l)

    if logits:
        logit_tensor = model(input_ph)

    model.add(Activation("softmax"))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    if logits:
        return model, logit_tensor

    return model


def evaluate(X_train, X_test, y_train, y_test):
    """Model  evaluation  function"""
    eval_params = {'batch_size': FLAGS.batch_size}
    train_acc = model_eval(sess, x, y, predictions,
                           X_train, y_train, args=eval_params)
    test_acc = model_eval(sess, x, y, predictions,
                          X_test, y_test, args=eval_params)
    print('Train acc: {:.2f} Test  acc: {:.2f} '.format(train_acc, test_acc))


def model_pred(sess, x, predictions, samples):
    feed_dict = {x: samples}
    probabilities = sess.run(predictions, feed_dict)

    print(probabilities, "************")

    if samples.shape[0] == 1:
        return np.argmax(probabilities)
    else:
        return np.argmax(probabilities, axis=1)


def generate_adv_samples(samples_to_perturb, jsma_params, X_test, y_test):
    adversarial_samples = []
    samples_perturbed_idxs = []

    for i, sample_ind in enumerate(samples_to_perturb):
        sample = X_test[sample_ind: sample_ind+1]

        # Finding adversarial example for all types of classes
        current_class = int(np.argmax(y_test[sample_ind]))
        target = 1 - current_class

        # Running the Jacobian-based saliency map
        one_hot_target = np.zeros((1, nb_classes), dtype=np.float32)
        one_hot_target[0, target] = 1
        jsma_params['y_target'] = one_hot_target

        # generating and storing the adversarial examples
        adv_x = jsma.generate_np(sample, **jsma_params)
        adversarial_samples.append(adv_x)
        samples_perturbed_idxs.append(sample_ind)

        # Check if the adversarial example was successful in fooling the model
        adv_tgt = np.zeros((1, FLAGS.nb_classes))
        adv_tgt[:, target] = 1
        res = int(model_eval(sess, x, y, predictions,
                  adv_x, adv_tgt, args={'batch_size': 1}))

        # Compute number of modified features
        adv_x_reshape = adv_x.reshape(-1)
        test_in_reshape = X_test[sample_ind].reshape(-1)
        nb_changed = np.where(adv_x_reshape != test_in_reshape)[0].shape[0]
        percent_perturb = float(nb_changed) / adv_x.reshape(-1).shape[0]

        results[target, sample_ind] = res
        perturbations[target, sample_ind] = percent_perturb

    malicious_targets = np.zeros((len(adversarial_samples), 2))
    malicious_targets[:, 1] = 1

    adversarial_samples = np.stack(adversarial_samples).squeeze()

    return adversarial_samples


if __name__ == '__main__':

    word2vec_seq_path = 'dataset/seq.csv'

    headers, X_train, X_test, y_train, y_test = prepare_data(word2vec_seq_path)

    print(classification_report(y_test, classify(X_train, y_train, X_test, y_test)))

    Sequential = keras.models.Sequential
    Dense = keras.layers.Dense
    Dropout = keras.layers.Dropout
    Activation = keras.layers.Activation

    plt.style.use('bmh')
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_integer(
        'nb_epochs', 1, 'Number  of  epochs  to train  model')  # was 20
    flags.DEFINE_integer(
        'batch_size', 256, 'Size of  training  batches ')  # was 32
    flags.DEFINE_float('learning_rate', 0.1, 'Learning  rate  for  training ')
    flags.DEFINE_integer(
        'nb_classes', y_train.shape[1], 'Number  of  classification  classes ')
    flags.DEFINE_integer(
        'source_samples', X_train.shape[1], 'Nb of test  set  examples  to  attack ')

    FLAGS = flags.FLAGS

    tf.compat.v1.flags.DEFINE_string('f', '', '')

    x = tf.compat.v1.placeholder(tf.float32, shape=(None, X_train.shape[1]))
    y = tf.compat.v1.placeholder(tf.float32, shape=(None, FLAGS.nb_classes))
    tf.compat.v1.set_random_seed(42)
    model = mlp_model((None, X_train.shape[1]))

    sess = tf.Session()
    keras.backend.set_session(sess)

    predictions = model(x)
    init = tf.global_variables_initializer()
    sess.run(init)

    # Train  the  model
    train_params = {
        'nb_epochs': FLAGS.nb_epochs,
        'batch_size': FLAGS.batch_size,
        'learning_rate': FLAGS.learning_rate,
        'verbose': 0}

    model_train(sess, x, y, predictions, X_train, y_train,
                evaluate=evaluate(X_train, X_test, y_train, y_test), args=train_params)

    # Generate  adversarial  samples  for  all  test  datapoints
    source_samples = X_test.shape[0]

    # Creating the Jacobian -based  Saliency  Map
    results = np.zeros((FLAGS.nb_classes, source_samples), dtype='i')
    perturbations = np.zeros((FLAGS.nb_classes, source_samples), dtype='f')
    grads = jacobian_graph(predictions, x, FLAGS.nb_classes)

    X_adv = np.zeros((source_samples, X_test.shape[1]))

    wrap = KerasModelWrapper(model)

    # Loop over the samples we want to perturb into adversarial examples
    samples_to_perturb = np.where(y_test[:, 1] == 1)[0]
    nb_classes = 2

    gamma = []
    theta = []

    for i in range(1, 10):
        gamma.append(i/10)
        theta.append(i/10)

    combinations = list(itertools.product(gamma, theta))

    jsma = SaliencyMapMethod(wrap, sess=sess)

    final_results = []

    for i in combinations:
        jsma_params = {'theta': i[1], 'gamma': i[0],
                       'clip_min': 0., 'clip_max': 1., 'y_target': None}
        adversarial_samples = generate_adv_samples(
            samples_to_perturb, jsma_params, X_test, y_test)
        adv_test = pd.DataFrame(adversarial_samples, columns=headers)

        adv_test['Result_False'] = 0
        adv_test['Result_True'] = 1

        test = pd.DataFrame(X_test, columns=headers)
        test['Result_False'] = y_test[:, 0]
        test['Result_True'] = y_test[:, 1]

        not_spam = test[test['Result_False'] == 1]

        joined = not_spam.append(adv_test, ignore_index=True)

        X_test_adv = np.array(joined[headers])
        y_test_adv = np.array(joined[['Result_False', 'Result_True']])

        final_results.append(f1_score(y_test, classify(
            X_train, y_train, X_test_adv, y_test_adv), average='weighted'))

        print(classification_report(y_test, classify(
            X_train, y_train, X_test_adv, y_test_adv)))

    results = pd.DataFrame(combinations, columns=['Gamma', 'Theta'])
    results['f1_score'] = final_results
    print(results)

    heatmap1_data = pd.pivot_table(results, values='f1_score', index=[
        'Gamma'], columns='Theta')

    fig, ax = plt.subplots(figsize=(20, 14))
    ax = sns.heatmap(heatmap1_data, annot=True, ax=ax)
    ax.invert_yaxis()
    # plt.savefig('heatmap.png')
