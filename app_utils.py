# Util functions for scraping urls and analyzing news texts with LIME for reusable, modular functionality in the app

# Quick start BS4: https://www.crummy.com/software/BeautifulSoup/bs4/doc/#
# Reference: https://brightdata.com/faqs/beautifulsoup/extract-text-from-html
import requests
import streamlit as st
from bs4 import BeautifulSoup
# Imports the custom explainability functions from lime_functions.py for generating LIME explanations
from lime_functions import explainPredictionWithLIME, displayAnalysisResults
from langdetect import detect, LangDetectException

def scrapeWithSoup(url):
    """"
    Creates a BeautifulSoup4 URL scraping back-up option if newspaper3k does not work.

            Input Parameters:
                url (str): the URL to scrape news content from

            Output:
                text (str): the news text to get out of the web page
    """


    # Creates scraping browser headers to circumvent being blocked by the news websites
    # Reference for creating request headers: https://stackoverflow.com/questions/72484699/scrap-image-with-request-header-on-beautifulsoup
    headers = {"User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:99.0) Gecko/20100101 Firefox/99.0", "cookie": "CONSENT=YES+cb.20230531-04-p0.en+FX+908"}

    # Loads in the HTML using requests library
    response = requests.get(url, headers=headers)

    # Extracts the HTML from the URL response
    html_content = response.text

    # Creates the Beautiful Soup object
    soup = BeautifulSoup(html_content,  "html.parser")

    # Extracts all the text from the HTML, this post was very helpful for resolving the scraping problem from many websites!
    # Reference: https://stackoverflow.com/questions/64691432/scrape-news-article-from-scraped-link-from-news-website-beautifulsoup-python
    text = soup.find_all("div", {"class", "most__list clearfix"})

    return text


def analyzeNewsText(news_text, fasttext_model, pipeline, scaler, feature_extractor, num_perturbed_samples, FEATURE_EXPLANATIONS, num_features=50):
    """"
    Reusable function for analyzing the news text (pasted or extracted from URL) with LIME and showing the
    visualizations, text, and highlighted text in the Streamlit app.

        Input Parameters:
            news_text (str): the news text to analyze
            fasttext_model (fasttext.FastText._FastText): the pre-trained FastText model
            pipeline (sklearn.pipeline.Pipeline): the pre-trained classifier model with Passive-Aggressive Classifier and CalibratedClassifierCV for outputting probabilities
            scaler (sklearn.preprocessing.StandardScaler): the pre-fitted StandardScaler
            feature_extractor (instance of custom made BasicFeatureExtractor): an instance of the class for extracting extra semantic and linguistic features
            num_perturbed_samples (int): user-selected number of perturbed samples for LIME to generate
            FEATURE_EXPLANATIONS (dict): maps extra feature column names to user-focused explanations
            num_features (int): number of word features for LIME Text Explainer to generate importance scores for, default = 50
    """

    # Displays the scraped original text in an expandable container
    with st.expander("View the Original News Text"):
        st.text_area("Original News Text", news_text, height=300)   

    # Generates the prediction and LIME explanation using the custom func in lime_functions.py
    with st.spinner("Analyzing text..."):
        explanation_dict = explainPredictionWithLIME(
            fasttext_model, # The FastText model
            pipeline, # The Passive-Aggressive Classifer model wrapped in CalibratedClassifierCV
            scaler, # The StandardScaler pre-fitted on the all-four training dat
            news_text, # The user-inputted news text
            feature_extractor, # An instance of the custom feature extractor instance for engineered features
            num_features=num_features, # The number of word features LIME generates with importance scores
            num_perturbed_samples=num_perturbed_samples # User-inputted num of perturbed samples for LIME to generate
        )

        # Displays the highlighted text, charts and LIME explanations
        displayAnalysisResults(explanation_dict, st, news_text, feature_extractor, FEATURE_EXPLANATIONS)


def detectIfTextIsEnglish(news_text):
    """"
    Returns False if text not in English, otherwise True.

        Input Parameters:
            news_text (str): the news text to analyze

        Output:
            True = text is in English
            False = text is not in English
            None = text language could not be detected
    """
    # Tries to detect whether text is in English
    try:
        language = detect(news_text)
        # Non-English text: return False
        if language != "en":
            return False
        else:
            return True # Text is in English
    # Not able to detect, returns None
    except Exception as e:
        return None