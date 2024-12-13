import re
import pickle
import pandas as pd
import base64
import matplotlib.pyplot as plt
from io import BytesIO
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Set stopwords and initialize the stemmer
STOPWORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Load the models and transformers
predictor = pickle.load(open("Models/model_xgb.pkl", "rb"))
scaler = pickle.load(open("Models/scaler.pkl", "rb"))
cv = pickle.load(open("Models/countVectorizer.pkl", "rb"))

# Streamlit UI setup
st.title("Sentiment Analysis")
option = st.sidebar.selectbox("Choose Input Type", ("Text Input", "Bulk CSV"))

# Function for single prediction
def single_prediction(predictor, scaler, cv, text_input):
    corpus = []

    # Preprocessing: cleaning and stemming the text
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
    review = " ".join(review)

    corpus.append(review)

    # Convert review into feature vector
    X_prediction = cv.transform(corpus).toarray()

    # Scaling the features
    X_prediction_scl = scaler.transform(X_prediction)

    # Get prediction probabilities
    y_predictions = predictor.predict_proba(X_prediction_scl)

    # Get the predicted class
    y_predictions = y_predictions.argmax(axis=1)[0]

    return "Positive" if y_predictions == 1 else "Negative"

# Function for bulk prediction
def bulk_prediction(predictor, scaler, cv, data):
    corpus = []

    for i in range(0, data.shape[0]):
        review = re.sub("[^a-zA-Z]", " ", data.iloc[i]["Sentence"])
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
        review = " ".join(review)
        corpus.append(review)

    # Convert reviews to feature vectors
    X_prediction = cv.transform(corpus).toarray()

    # Scaling the features
    X_prediction_scl = scaler.transform(X_prediction)

    # Get prediction probabilities
    y_predictions = predictor.predict_proba(X_prediction_scl)

    # Get the predicted classes
    y_predictions = y_predictions.argmax(axis=1)
    y_predictions = list(map(sentiment_mapping, y_predictions))

    data["Predicted sentiment"] = y_predictions
    predictions_csv = BytesIO()

    data.to_csv(predictions_csv, index=False)
    predictions_csv.seek(0)

    graph = get_distribution_graph(data)

    return predictions_csv, graph

# Function for sentiment mapping
def sentiment_mapping(x):
    if x == 1:
        return "Positive"
    else:
        return "Negative"

# Function for generating distribution graph
def get_distribution_graph(data):
    fig = plt.figure(figsize=(5, 5))
    colors = ("green", "red")
    wp = {"linewidth": 1, "edgecolor": "black"}
    tags = data["Predicted sentiment"].value_counts()
    explode = (0.01, 0.01)

    tags.plot(
        kind="pie",
        autopct="%1.1f%%",
        shadow=True,
        colors=colors,
        startangle=90,
        wedgeprops=wp,
        explode=explode,
        title="Sentiment Distribution",
        xlabel="",
        ylabel="",
    )

    graph = BytesIO()
    plt.savefig(graph, format="png")
    plt.close()

    return graph

# Main logic for text input or file upload
if option == "Text Input":
    text_input = st.text_area("Enter Text for Sentiment Analysis")
    
    if text_input:
        predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)
        st.write(f"Predicted Sentiment: {predicted_sentiment}")

elif option == "Bulk CSV":
    file = st.file_uploader("Upload CSV file with 'Sentence' column", type=["csv"])

    if file:
        data = pd.read_csv(file)

        if "Sentence" in data.columns:
            predictions_csv, graph = bulk_prediction(predictor, scaler, cv, data)

            st.write("Bulk Predictions:")
            st.write(data)

            # Download link for predictions CSV
            st.download_button(
                label="Download Predictions",
                data=predictions_csv,
                file_name="predictions.csv",
                mime="text/csv",
            )

            # Display sentiment distribution graph
            st.image(graph)

        else:
            st.write("The uploaded CSV must contain a 'Sentence' column.")
