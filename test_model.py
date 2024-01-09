import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from urllib.parse import urlparse
import ipaddress
import re
import sys

def load_dataset():
    # Check if data is provided through standard input
    # if not sys.stdin.isatty():
    #     dataset_buffer = sys.stdin.buffer.read()
    #     dataset = pd.read_csv(pd.compat.StringIO(dataset_buffer.decode('utf-8')))
    # else:
        # If no input is provided through standard input, load data from file
    dataset = pd.read_csv('dataset_phishings.csv')
    return dataset

def preprocess_original_dataset(dataset):
    raw_dataset = dataset.copy()
    raw_dataset.head()
    raw_dataset.shape
    raw_dataset.describe()
    pd.set_option('display.max_rows', 500)
    raw_dataset.isna().sum()
    pd.reset_option('display.max_rows')
    original_dataset = raw_dataset.copy()
    
    class_mapping = {'legitimate': 0, 'phishing': 1}
    original_dataset['status'] = original_dataset['status'].map(class_mapping)

    numeric_columns = original_dataset.select_dtypes(include=[np.number]).columns
    corr_matrix = original_dataset[numeric_columns].corr()
    
    status_corr = corr_matrix['status']

    features_selected = feature_selector_correlation(status_corr, 0.2)
    selected_features = [feature for (feature, score) in features_selected if feature != 'status']
    X_selected = original_dataset[selected_features]
    
    y = original_dataset['status']

    return X_selected, y

def train_model(X_train, y_train):
    model_random_forest = RandomForestClassifier(n_estimators=350, random_state=42)
    model_random_forest.fit(X_train, y_train)
    return model_random_forest

def visualize_distribution(original_dataset):
    counts = original_dataset['status'].value_counts()
    labels = counts.index
    sizes = counts.values

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['lightblue', 'lightcoral'])
    plt.title('Distribution of Legitimate and Phishing URLs')
    plt.show()

def custom_accuracy_set(model, X_train, X_test, y_train, y_test, train=True):
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_train)
    
    if train:
        x = X_train
        y = y_train
    else:
        x = X_test
        y = y_test
        
    y_predicted = model.predict(x)
    
    accuracy = accuracy_score(y, y_predicted)
    oconfusion_matrix = confusion_matrix(y, y_predicted)
    oroc_auc_score = lb.transform(y), lb.transform(y_predicted)

def save_model(model, filename='trained_model.pkl'):
    with open(filename, 'wb') as model_file:
        pickle.dump(model, model_file)

def load_model(filename='trained_model.pkl'):
    with open(filename, 'rb') as model_file:
        model = pickle.load(model_file)
    return model

def main():
    dataset = load_dataset()
    X_selected, y = preprocess_original_dataset(dataset)
    
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, shuffle=True)
    
    model_random_forest = train_model(X_train, y_train)
    
    visualize_distribution(dataset)
    
    custom_accuracy_set(model_random_forest, X_train, X_test, y_train, y_test, train=True)
    custom_accuracy_set(model_random_forest, X_train, X_test, y_train, y_test, train=False)

    save_model(model_random_forest)

    model_random_forest = load_model()

    test_links = ["http://facebook.com"]
    test_data = preprocess_test_data(test_links, X_selected.columns)
    predicted_labels = model_random_forest.predict(test_data)

    for url, label in zip(test_links, predicted_labels):
        print(f"URL: {url} - Predicted Label: {'Legitimate' if label == 0 else 'Phishing'}")
        sys.stdout.write(f"{label}\n")
        sys.stdout.flush()

    predicted_counts = pd.Series(predicted_labels).value_counts()
    labels = ['Legitimate', 'Phishing']
    sizes = [predicted_counts.get(0, 0), predicted_counts.get(1, 0)]

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['lightblue', 'lightcoral'])
    plt.title('Distribution of Predicted Labels for Test URLs')
    plt.show()

if __name__ == "__main__":
    main()