"""
Purpose: Classification of MOOC forum posts based on their urgency from 1 (not urgent at all) to 7 (extremely urgent).
Author: Valdemar Švábenský <svabenskyv@gmail.com>, University of Pennsylvania, 2022--2023.
Reviewed by: Andrés Felipe Zambrano, Stefan Slater

Configuration on which the code was tested: Python 3.10 on Windows 10 (with 16 GB of RAM).
Approximate total computation time for the best approach (SVM with USE) is 45 seconds.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Text data cleaning and preprocessing
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# BOW and TF-IDF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Classification
from xgboost import XGBClassifier  # Requires version 1.5.0 for y values that range starting from 1 and not 0
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

# Regression
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from mord import OrdinalRidge

# Cross-validation and other helper modules
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# Performance evaluation metrics
from sklearn.metrics import mean_squared_error, roc_auc_score, f1_score, classification_report
from scipy.stats import spearmanr
from statistics import mean, stdev

# Universal Sentence Encoder and neural networks
import tensorflow as tf
import tensorflow_hub as hub


# ===== DATA CLEANING AND PREPROCESSING =====

unwanted_words = set(stopwords.words('english'))
unwanted_words.add('redacted')  # Raw data contain this word because of anonymization of student names
stemmer = PorterStemmer()  # Performs tiny bit better than lemmatiser = WordNetLemmatizer()


def normalize_post_text(string):
    """
    Convert the forum post to lowercase, replace all non-alphanumeric characters, then
    remove stopwords/unwanted words, and either stem or lemmatize.
    (See Section 3.4 in the paper.)
    :param string (str). Forum post text.
    :return: (str). The normalized post text.
    """
    string = string.lower()
    string = re.sub(r'[^a-z0-9]', ' ', string)  # We want to keep only letters and numbers in the text data
    all_words = string.split()
    wanted_words = [word for word in all_words if word not in unwanted_words]
    stemmed_words = [stemmer.stem(word) for word in wanted_words]
    new_string = ' '.join(stemmed_words)
    return new_string


def prepare_data(filepath='All_Courses_REDACTED_CODED.csv'):
    """
    Read the input CSV file, normalize the text data, and prepare the variables for analysis.
    :param filepath (str). Path to the input file.
    :return: (DataFrame, DataFrame). X vector of features and y labels from 1 to 7 (ordinal).
    """
    df = pd.read_csv(filepath)
    df['post_text'] = df['post_text'].apply(normalize_post_text)
    # df.info()  # post_text is described as "object", but is actually "str"; everything is non-null
    X = df.copy()
    y = X.pop('Urgency_1_7').apply(float)  # Do not .apply(round) here to preserve .5 values in the Stanford dataset

    assert all(1 <= label <= 7 for label in y)
    if BINARY_CLASSIFICATION:  # Convert the labels to binary (1--4 becomes 0, 4.5--7 becomes 1)
        y = y.apply(lambda label: 1 if label > 4 else 0)
    print(y.describe())

    return X, y


# ===== HELPER FUNCTIONS FOR MODEL TRAINING AND CROSS-VALIDATION -- USED IN ALL METHODS LATER BELOW =====

def create_student_cv_groups(X):
    """
    Code by Miggy Andres-Bray (https://www.miggyandresbray.com/) to be able to achieve student-level cross-validation
    during model training.

    We create a NumPy array `groups` of size equal to the number of rows in the input dataset `X`,
    such that each element of `groups` is the index of the first occurrence of a student's ID at that place in `X`.

    Example: Assume these student IDs in the input dataset: [A, B, A, C, B]
    Expected output (result of `groups`): [0. 1. 0. 3. 1]

    :param X (DataFrame). Dataframe of features that also include student ID.
    :return: (np.array). Array of IDs showing which data points belong to the same students.
    """
    group_dict = {}
    groups = np.array([])
    for index, row in X.iterrows():
        student_id = int(row['id'])
        if student_id not in group_dict:
            group_dict[student_id] = index
        groups = np.append(groups, group_dict[student_id])
    return groups


def get_train_test_split(X, y, train_index, test_index):
    """ Split the data into the train and test set based on the provided index. """
    if type(X) is np.ndarray:  # This applies for word-count (WC) based models (BOW and TF-IDF)
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
    elif issubclass(type(X), tf.Tensor):  # This applies for using TensorFlow (USE)
        # Source: https://stackoverflow.com/questions/46092394/slicing-tensor-with-int32-shape-with-int64-scalar
        train_index_tf = tf.Variable(train_index, dtype=tf.int64)
        test_index_tf = tf.Variable(test_index, dtype=tf.int64)
        X_train = tf.gather(X, train_index_tf)
        X_test = tf.gather(X, test_index_tf)
        y_train = tf.gather(y, train_index_tf)
        y_test = tf.gather(y, test_index_tf)
    else:
        raise TypeError('X must be a NumPy array or a Tensor matrix.')
    return X_train, X_test, y_train, y_test


def test_model_performance(X_test, y_test, model, visualize=False):
    """ Fit the model and evaluate its performance on the given test set. """
    if not BINARY_CLASSIFICATION:
        y_test_predicted = model.predict(X_test)  # Use for all regressors (incl. NNs) and classifiers with the default threshold
        result_metrics = [mean_squared_error(y_test, y_test_predicted, squared=False),
                          spearmanr(y_test, y_test_predicted).correlation]
    else:
        if type(model) is tf.keras.Sequential:
            y_test_predicted = np.round(model.predict(X_test))  # Use for NN classification
        else:
            y_test_predicted = model.predict(X_test)
            # y_test_predicted = (model.predict_proba(X_test)[:, 1] >= 0.2).astype(bool)  # Here you can set the decision threshold cut-off

        per_class_metrics = classification_report(y_test, y_test_predicted, output_dict=True)
        result_metrics = [roc_auc_score(y_test, y_test_predicted),
                          f1_score(y_test, y_test_predicted, average='macro'),
                          f1_score(y_test, y_test_predicted, average='weighted'),
                          per_class_metrics['0']['f1-score'],
                          per_class_metrics['1']['f1-score']]

    if visualize and not BINARY_CLASSIFICATION:
        plot_model(y_test, y_test_predicted)

    return result_metrics


def train_and_cross_validate_model(X, y, groups, model, n_splits=10):
    """ Using the functions above, train and cross-validate the given model. """
    metrics_values = [[] for _ in range(5)]  # Individual (= per each fold) RMSE and Spearman values *OR* AUC and F1 values
    gkf = GroupKFold(n_splits=n_splits)
    for train_index, validate_index in gkf.split(X, y, groups=groups):
        X_train, X_validate, y_train, y_validate = get_train_test_split(X, y, train_index, validate_index)
        if type(model) is tf.keras.Sequential:
            model = prepare_and_fit_NN_model(X_train, y_train)  # Keras model objects must be created newly in each iteration
            _, performance = model.evaluate(X_validate, y_validate)  # Check for overfit happening
        else:
            model.fit(X_train, y_train)  # Sk-learn model objects are always re-fitted freshly in-place
        results = test_model_performance(X_validate, y_validate, model)
        for i in range(len(results)):
            metrics_values[i].append(results[i])

    print(model)  # Print the cross-validation results
    for metric_values in metrics_values:
        if metric_values:
            print(mean(metric_values))

    if type(model) is not tf.keras.Sequential:
        model.fit(X, y)  # Fit the final Sk-learn model on the whole training set (Keras models are discarded)


def plot_model(y_test, y_test_predicted):
    possible_labels = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]  # For the test set
    predicted_values_per_label = {label: [] for label in possible_labels}  # List of values for each urgency label
    for i in range(len(y_test)):
        predicted_values_per_label[y_test[i]].append(y_test_predicted[i])  # All predicted values for each urgency label
    avg_predicted_values_per_label = [mean(predicted_values_per_label[x]) for x in possible_labels]
    stdev_predicted_values_per_label = [stdev(predicted_values_per_label[x]) for x in possible_labels]

    plt.axis('square')  # Make the x and y scales equal
    plt.xlim(0.8, 7.4)
    plt.ylim(0.8, 4.5)
    plt.yticks([1, 2, 3, 4])
    plt.xlabel("Actual urgency label")
    plt.ylabel("Predicted urgency label (mean ± SD)")

    # plt.plot(possible_labels, avg_predicted_values_per_label, linestyle='--', marker='o', label='Average')
    # plt.plot(possible_labels, stdev_predicted_values_per_label, linestyle=':', marker='s', label='Standard deviation')
    # plt.legend()
    plt.errorbar(possible_labels, avg_predicted_values_per_label, stdev_predicted_values_per_label, linestyle='', marker='o', capsize=2)
    offset = 0.05
    for i in range(len(possible_labels)):
        x_text_pos = possible_labels[i] + offset
        y_value_avg = avg_predicted_values_per_label[i]
        y_value_stdev = stdev_predicted_values_per_label[i]
        y_low_point, y_high_point = y_value_avg - y_value_stdev, y_value_avg + y_value_stdev

        plt.text(x=x_text_pos, y=y_value_avg + offset, s=f"{round(y_value_avg, 1)}", fontsize='x-small')
        plt.text(x=x_text_pos, y=y_low_point + offset, s=f"{round(y_low_point, 1)}", fontsize='xx-small', color='gray')
        plt.text(x=x_text_pos, y=y_high_point + offset, s=f"{round(y_high_point, 1)}", fontsize='xx-small', color='gray')
    # plt.show(); quit()
    plt.savefig('plot.png', dpi=1200, bbox_inches='tight')


# ===== (METHOD 1) SIMPLE WORD COUNTS (WC) USING BAG OF WORDS OR TF-IDF =====

def prepare_WC_feature_vectors(X, X_Stanford, method='tf-idf', ngram_range=(1, 1)):
    """
    :param X (DataFrame). Vector imported from the raw training data.
    :param X_Stanford (DataFrame). Vector imported from the raw testing data.
    :param method (str). 'bow' for Bag of Words or 'tf-idf'.
        The former is faster, but the latter tends to be more precise.
    :param ngram_range (int, int). A tuple of how many n-grams to consider.
        (1, 1) is the fastest. (1, 2) is sometimes more accurate. (2, 2) is always worse.
    :return: (DataFrame). Feature vector based on word count.
    """
    assert method in ('bow', 'tf-idf')

    # Training and cross-validation
    if method == 'bow':
        matrix_TF = CountVectorizer(ngram_range=ngram_range, min_df=0.01, max_df=0.99)  # Higher values seem to favor regression models
    else:
        matrix_TF = TfidfVectorizer(ngram_range=ngram_range, min_df=0.01, max_df=0.99)  # Set experimentally
    post_text_data = X['post_text'].tolist()  # From dataframe column to list of strings
    X_WC = matrix_TF.fit_transform(post_text_data).toarray()

    # Descriptive statistics of the post texts
    post_text_lengths = [len(post_text.split()) for post_text in post_text_data]
    post_text_lengths.remove(0)
    # print(pd.DataFrame(post_text_lengths).describe())

    # Testing on the holdout test set
    if method == 'bow':
        matrix_TF_Stanford = CountVectorizer(ngram_range=ngram_range, vocabulary=matrix_TF.vocabulary_)
    else:
        matrix_TF_Stanford = TfidfVectorizer(ngram_range=ngram_range, vocabulary=matrix_TF.vocabulary_)
    post_text_data_Stanford = X_Stanford['post_text'].tolist()
    X_WC_Stanford = matrix_TF_Stanford.fit_transform(post_text_data_Stanford).toarray()

    # Optional, to see the features (words)
    # df_bow = pd.DataFrame(X_WC_Stanford, columns=matrix_TF.get_feature_names_out())
    # print(df_bow.head())

    return X_WC, X_WC_Stanford


# ===== (METHOD 2) UNIVERSAL SENTENCE ENCODER WITHOUT NEURAL NETWORKS =====

def prepare_USE_feature_vectors(X, X_Stanford):
    """
    :param X (DataFrame). Vector imported from the raw training data.
    :param X_Stanford (DataFrame). Vector imported from the raw testing data.
    :return: (Tensor). Matrix of embeddings for each post test.
    """
    # module_url = 'https://tfhub.dev/google/universal-sentence-encoder/4'  # online version
    module_url = '../universal-sentence-encoder_4'  # download v4 local if remote handling from URL crashes
    # module_url = '../universal-sentence-encoder-large_5'  # v5 embeddings are too big to fit in memory, it crashes
    embed = hub.load(module_url)
    X_USE = embed(X['post_text'])
    X_USE_Stanford = embed(X_Stanford['post_text'])
    return X_USE, X_USE_Stanford


# ===== (METHOD 3) NEURAL NETWORKS =====

def prepare_and_fit_NN_model(X_train, y_train):
    """ Get a completely new Keras model and fit it on the training set. """
    model = tf.keras.Sequential()
    input_shape = len(X_train[0])  # 774 for WC and 512 for USE
    if not BINARY_CLASSIFICATION:  # Default case, regression NN model
        model.add(tf.keras.layers.Dense(128, name='input', input_shape=(input_shape,), activation='relu'))
        model.add(tf.keras.layers.Dropout(0.85, input_shape=(128,)))
        model.add(tf.keras.layers.Dense(128, name='hidden1', activation='relu'))
        model.add(tf.keras.layers.Dropout(0.85, input_shape=(128,)))
        # model.add(tf.keras.layers.Dense(8, name='hidden2', activation='relu'))
        # model.add(tf.keras.layers.Dense(8, name='output', activation='softmax'))  # Classification: number of classes + 1
        model.add(tf.keras.layers.Dense(1, name='output', activation='relu'))
        model.compile(loss='mean_squared_error',
                      optimizer='adam',  # Generally the best default
                      metrics=[tf.keras.metrics.RootMeanSquaredError()])
    else:
        model.add(tf.keras.layers.Dense(128, name='input', input_shape=(input_shape,), activation='tanh'))
        model.add(tf.keras.layers.Dropout(0.85, input_shape=(128,)))
        model.add(tf.keras.layers.Dense(1, name='output', activation='sigmoid'))  # softmax for multiclass classification
        model.compile(loss='binary_crossentropy',  # sparse_categorical_crossentropy for multiclass classification
                      optimizer='adam',
                      metrics=[tf.keras.metrics.AUC()])

    # batch_size = how many samples to use, should be a multiple of 32
    # epochs = number of iterations in which the parameters will be calculated until all data are used
    model.fit(X_train, y_train, batch_size=128, epochs=64, verbose=0)
    return model


# ===== MAIN =====

def main(method):
    X, y = prepare_data('All_Courses_REDACTED_CODED.csv')  # Training set
    groups = create_student_cv_groups(X)
    X_Stanford, y_Stanford = prepare_data('Stanford.csv')  # Test set

    assert method in ('WC', 'USE')
    if method == 'WC':
        X_train, X_test = prepare_WC_feature_vectors(X, X_Stanford)
    else:
        X_train, X_test = prepare_USE_feature_vectors(X, X_Stanford)
    # X_train = StandardScaler().fit_transform(X_train)
    # X_test = StandardScaler().fit_transform(X_test)

    modelCART = tree.DecisionTreeClassifier()
    modelRF = RandomForestClassifier()
    modelXGB = XGBClassifier()
    modelLinReg = LinearRegression()
    modelOrdReg = OrdinalRidge()
    modelSVReg = SVR()
    modelNN = tf.keras.Sequential()
    for model in [modelSVReg]:  # modelCART, modelRF, modelXGB, modelLinReg, modelOrdReg, modelSVReg, modelNN
        train_and_cross_validate_model(X_train, y, groups, model)
        if model == modelNN:
            model = prepare_and_fit_NN_model(X_train, y)  # Keras models must be fitted again, the objects are lost inside the function
        print(test_model_performance(X_test, y_Stanford, model, True))  # Add True here to create plots


if __name__ == '__main__':
    BINARY_CLASSIFICATION = False  # Switch depending on your needs
    main('WC')  # Set to 'WC' or 'USE' depending on your needs
