import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  # Import NumPy for numeric operations
from urllib.parse import urlparse,urlencode # import more library for the trained model
import ipaddress
import re
import sys


def main():
   
     # Retrieve command line arguments (url and filePath)
    url = sys.argv[1]
    filePath = sys.argv[2]
    # dataset = pd.read_csv('dataset_phishing.csv')
    dataset = pd.read_csv(filePath)



    raw_dataset = dataset.copy()
    raw_dataset.head()

    raw_dataset.shape
    raw_dataset.describe()
    pd.set_option('display.max_rows', 500)
    raw_dataset.isna().sum()
    pd.reset_option('display.max_rows')
    original_dataset = raw_dataset.copy()

    class_map = {'legitimate': 0, 'phishing': 1}
    original_dataset['status'] = original_dataset['status'].map(class_map)

    # Filter numeric columns only
    numeric_columns = original_dataset.select_dtypes(include=[np.number]).columns

    # # Calculate correlation matrix for numeric columns only
    corr_matrix = original_dataset[numeric_columns].corr()

    corr_matrix.shape
    corr_matrix['status']
    status_corr = corr_matrix['status']
    status_corr.shape

    def feature_selector_correlation(cmatrix, threshold):
        
        selected_features = []
        feature_score = []
        i=0
        for score in cmatrix:
            if abs(score)>threshold:
                selected_features.append(cmatrix.index[i])
                feature_score.append( ['{:3f}'.format(score)])
            i+=1
        result = list(zip(selected_features,feature_score)) 
        return result

    features_selected = feature_selector_correlation(status_corr, 0.2)

    selected_features = [i for (i,j) in features_selected if i != 'status']

    X_selected = original_dataset[selected_features]

    X_selected.shape
    y = original_dataset['status']

    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, shuffle = True)

    model_random_forest = RandomForestClassifier(n_estimators=350, random_state=42,)
    model_random_forest.fit(X_train,y_train)

    # Assuming 'original_dataset' is your DataFrame containing the 'status' column
    counts = original_dataset['status'].value_counts()
    labels = counts.index
    sizes = counts.values

    # Plotting the pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['lightblue', 'lightcoral'])
    plt.title('Distribution of Legitimate and Phishing URLs')
    plt.show()


    def custom_accuracy_set (model, X_train, X_test, y_train, y_test, train=True):
        
        lb = preprocessing.LabelBinarizer()
        lb.fit(y_train)
        
        if train:
            x = X_train
            y = y_train
        elif not train:
            x = X_test
            y = y_test
            
        y_predicted = model.predict(x)
        
        accuracy = accuracy_score(y, y_predicted)
        # print('model accuracy: {0:4f}'.format(accuracy))
        oconfusion_matrix = confusion_matrix(y, y_predicted)
        # print('Confusion matrix: \n {}'.format(oconfusion_matrix))
        oroc_auc_score = lb.transform(y), lb.transform(y_predicted)


    custom_accuracy_set(model_random_forest, X_train, X_test, y_train, y_test, train=True)
    custom_accuracy_set(model_random_forest, X_train, X_test, y_train, y_test, train=False)

    with open('trained_model.pkl', 'wb') as model_file:
        pickle.dump(model_random_forest, model_file)

    # Load the trained model
    with open('trained_model.pkl', 'rb') as model_file:
        model_random_forest = pickle.load(model_file)
        
    # Define functions for additional features
    def havingIP(url):
        parsed_url = urlparse(url)
        try:
            ipaddress.ip_address(parsed_url.hostname)
            return 1
        except ValueError:
            return 0

    def haveAtSign(url):
        return 1 if "@" in url else 0

    def getLength(url):
        return 0 if len(url) < 54 else 1

    def getDepth(url):
        path_segments = urlparse(url).path.split('/')
        # Filter out empty segments
        path_segments = [segment for segment in path_segments if segment]
        depth = len(path_segments)
        return depth

    def httpDomain(url):
        domain = urlparse(url).netloc
        return 1 if 'https' in domain else 0

    shortening_services = r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|" \
                        r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|" \
                        r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|" \
                        r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|db\.tt|" \
                        r"qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|q\.gs|is\.gd|" \
                        r"po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|" \
                        r"prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|" \
                        r"tr\.im|link\.zip\.net"

    def tinyURL(url):
        match = re.search(shortening_services, url)
        return 1 if match else 0

    def extract_features(url):
        parsed_url = urlparse(url)

        # Basic URL components
        features = []
        features.append(havingIP(url))
        features.append(haveAtSign(url))
        features.append(getLength(url))
        features.append(getDepth(url))
        features.append(httpDomain(url))
        features.append(tinyURL(url))
        features.append(len(parsed_url.netloc))  # Domain length
        features.append(parsed_url.netloc.count('.') + 1)  # Number of subdomains
        features.append(len(parsed_url.path))  # Path length
        features.append(parsed_url.path.count('/') + 1)  # Number of path segments

        return features

    def preprocess_test_data(test_links, selected_features_order):
        test_features = [extract_features(url) for url in test_links]
        test_df = pd.DataFrame(test_features)

        # Handle missing values, if any
        test_df.fillna(0, inplace=True)  # Replace NaN with 0, adjust based on your preprocessing strategy

        # Ensure the order of features in 'test_df' matches the order used during training
        test_df = test_df.reindex(columns=selected_features_order, fill_value=0)

        return test_df


    # Example test links
    # url = "http://facebook.com"
    test_links = [url]

    # Preprocess the test data
    test_data = preprocess_test_data(test_links, X_selected.columns)

    # Predict the labels using the trained model
    predicted_labels = model_random_forest.predict(test_data)

    # Display the results
    for url, label in zip(test_links, predicted_labels):
        # print(f"URL: {url} - Predicted Label: {'Legitimate' if label == 0 else 'Phishing'}")
        print(f"{'Legitimate' if label == 0 else 'Phishing'}")
        # sys.stdout.write(f"{label}\n")
        # sys.stdout.flush()

    # Generate a pie chart to visualize the distribution of predicted labels
    predicted_counts = pd.Series(predicted_labels).value_counts()
    labels = ['Legitimate', 'Phishing']
    sizes = [predicted_counts.get(0, 0), predicted_counts.get(1, 0)]

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['lightblue', 'lightcoral'])
    plt.title('Distribution of Predicted Labels for Test URLs')
    plt.show()

if __name__ == "__main__":
    main()
 