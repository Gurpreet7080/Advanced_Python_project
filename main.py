import re
import pickle
import pandas as pd
import base64
import matplotlib.pyplot as plt
from io import BytesIO
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# nltk.download('stopwords')
STOPWORDS = set(stopwords.words("english"))

# Load the models and transformers
predictor = pickle.load(open(r"Models/model_xgb.pkl", "rb"))
scaler = pickle.load(open(r"Models/scaler.pkl", "rb"))
cv = pickle.load(open(r"Models/countVectorizer.pkl", "rb"))

# Title of the web app
st.title("Sentiment Analysis")

# Sidebar for file upload or text input
option = st.sidebar.selectbox("Choose Input Type", ("Text Input", "Bulk CSV"))

# Function for single prediction
def single_prediction(predictor, scaler, cv, text_input):
    corpus = []
    stemmer = PorterStemmer()

    # Preprocessing
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
    review = " ".join(review)
    
    st.write(f"Processed review: {review}")  # Print processed review text
    
    corpus.append(review)
    X_prediction = cv.transform(corpus).toarray()
    
    X_prediction_scl = scaler.transform(X_prediction)
    
    # Get prediction
    y_predictions = predictor.predict_proba(X_prediction_scl)
    
    # Get the predicted class
    y_predictions = y_predictions.argmax(axis=1)[0]
    
    return "Positive" if y_predictions == 1 else "Negative"

# Function for bulk predictions
def bulk_prediction(predictor, scaler, cv, data):
    corpus = []
    stemmer = PorterStemmer()

    for i in range(0, data.shape[0]):
        review = re.sub("[^a-zA-Z]", " ", data.iloc[i]["Sentence"])
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
        review = " ".join(review)
        corpus.append(review)

    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
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
    user_input = st.text_area("Enter text for sentiment prediction")

    if st.button("Predict Sentiment"):
        if user_input:
            # Call function to predict sentiment
            prediction = single_prediction(predictor, scaler, cv, user_input)
            st.subheader(f"Predicted Sentiment: {prediction}")
        else:
            st.error("Please enter some text.")

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
