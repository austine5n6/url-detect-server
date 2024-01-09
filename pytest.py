import pandas as pd
import pickle
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import re
import ipaddress


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

# Define a function to preprocess URLs and extract features
def preprocess_url(url):
    parsed_url = urlparse(url)

    # Extract features from the URL
    features = {
        'url_length': len(url),
        'hostname_length': len(parsed_url.hostname) if parsed_url.hostname else 0,
        'path_length': len(parsed_url.path),
        'number_of_dots': url.count('.'),
        'number_of_digits': sum(c.isdigit() for c in url),
        'having_IP': havingIP(url),
        'have_AtSign': haveAtSign(url),
        'URL_Length': getLength(url),
        'URL_Depth': getDepth(url),
        'https_Domain': httpDomain(url),
        'Tiny_URL': tinyURL(url),
    }

    return pd.DataFrame([features])

# Define a function to predict the status of a URL
def predict_url_status(url, model):
    # Preprocess the URL
    processed_url = preprocess_url(url)

    # Use the trained model to predict the status
    predicted_status = model.predict(processed_url)

    return predicted_status[0]

# Test URLs
test_urls = [
    'https://www.example.com',  # Legitimate URL
    'http://fake-phishing-site.com',  # Phishing URL
    'http://another-legitimate-site.org',  # Legitimate URL
]

# Predict the status for each test URL
for url in test_urls:
    status = predict_url_status(url, model_random_forest)
    print(f"URL: {url}, Predicted Status: {'Legitimate' if status == 0 else 'Phishing'}")