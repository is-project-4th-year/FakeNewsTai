# Functions for outputting LIME feature importance scores and visualizations

# Imports basic data processing libs
import pandas as pd
import numpy as np
# Imports RegEx for preprocessing texts
import re 
# Imports Streamlit for app creation
import streamlit as st
# Imports text processing libs + downloads stopwords
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
# Defines the global set of stopwords to remove important word feats that are stopwords
stop_words = set(stopwords.words("english"))
# Imports Altair for Streamlit data visualization/bar charts
# Reference: https://docs.streamlit.io/develop/api-reference/charts/st.altair_chart
import altair as alt
# Imports the LIME explanation library: class for text-based explanations
from lime.lime_text import LimeTextExplainer
# Imports the class for generating extra numeric feature score based explanations
from lime.lime_tabular import LimeTabularExplainer


def preprocessTextForFastTextEmbeddings(text):
    """
    - A basic text preprocessing function required before extracting the fastText dense embeddings.
    - It only cleans text from extra whitespace and newlines, as well as lowercasing,
    in preparation for training a fastText model, as this model cannot handle newlines and trailing spaces.
    -  It does not remove punctuation as based on feature analysis, this can be an important distinguishing factor for fake news,
    as shown before in the EDA.
    - It does not remove stopwords because they could be important to fastText model, which has been autotuned to poss. include
    n-grams (bi/trigrams), where stopwords can be important to determining meaning of phrasal verb patterns, collocations etc.
    - Also does not aggressively apply normalization (lemmatize/stemming) words as extraction and comparison of morphemes
    and subword components is automatically handled by the FastText model.
        
        Input Parameters:
            text (str): the text to preprocess
        Output:
            text (str): the cleaned text
    """
    # Converts the text to lowercase
    text = text.lower()
   
    # Removes the newlines which don't work with FastText models and extra whitespace (the \s regex matches any whitespace character including tabs/newlines)
    text = re.sub(r"\s+", ' ', text)

    # Strips the remaining trailing whitespace
    text = text.strip()
   
    return text


def getFastTextEmbedding(model, text):
    """
    Extracts the fastText dense embedding for a text using a pre-trained fastText model.
    
    Input Parameters:
        model: the pre-trained fastText model, trained on all-four combined text datasets
        text (str): the text to extract the embedding for
        
    Output:
        numpy.ndarray: the fastText dense embedding as a numpy array
    """
    # Cleans the text using the above helper function
    text = preprocessTextForFastTextEmbeddings(text)

    # Extracts the dense embedding for this text from the fastText model
    return model.get_sentence_vector(text)


def combineFeaturesForSingleTextDF(single_text_df, scaler, feature_cols):
    """
    Scales and combines extra engineered engineered features and concatenates them into a row with the fastText
    embeddings, outputting a 2D numpy array with a single row containing the concatenated features.

        Input Parameters:

        single_text_df (pd.DataFrame): a pandas DataFrame containing a single row with a text, fastText embedding, and extra feature scores
        scaler (sklearn.preprocessing.StandardScaler): a pre-fitted, saved StandardScaler instance for scaling the engineered features
        feature_cols (list): the list of column names for extracting the engineered features
        
        Output:
            final_vector (numpy.ndarray): the final combined embeddings + feature vector for inputting into the classifier model
    """

    # Extracts the fastText embedding from the single-text DataFrame (will be at row at index 0, as there is only 1 row here) 
    embedding = single_text_df["fasttext_embeddings"].iloc[0]

    # Checks if it a np array type, otherwise if was converted into string instead, extracts the values in between the
    # square brackets, uses space as a separator to identify each float, and converts to np array using .fromstring
    if not isinstance(embedding, np.ndarray):
        embedding = np.fromstring(embedding.strip("[]"), sep=" ")
        
    # Scales the extra features using the pre-fitted StandardScaler
    engineered_features = scaler.transform(single_text_df[feature_cols])

    # Gets the max. value from the engineered features to check if it is a val greater than 1, to see if scaled properly
    max_abs_value_features = np.max(np.abs(engineered_features))
    
    # If the max. feature value is greater than 1, then scale it down by a scaling factor
    if max_abs_value_features > 1:
        # Calculates the scaling factor to get all features under 1: 1 divided by value of max feature
        scale_factor = 1 / max_abs_value_features
        # Applies the scale-factor element-wise to the engineered features, to make them all less than absolute value of 1
        engineered_features = engineered_features * scale_factor

    # Adds a new dimension to the embedding array to make it 2D for model training
    embedding = embedding.reshape(1, -1) 

    # Concatenates the single-row containing the embedding with the engineered features for the final text
    final_vector = np.hstack([embedding, engineered_features])

    # Returns the embedding + scaled features vector for model predictions
    return final_vector


def explainPredictionWithLIME(
    fasttext_model,
    classifier_model,
    fitted_scaler,
    text,
    feature_extractor,
    num_features=50,
    num_perturbed_samples=100
):
    """
    - Extracts the features from a user-inputted news text, predicts the probability of the text being fake news by using a custom pipeline
    that first extracts FastText embeddings, extracts and scales the extra features, and inserts them into
    a pre-trained Passive-Aggressive classifier, wrapped in a Calibrated Classifier, for outputting fake vs real news probabilities.
    - It then uses LIME Text and Tabular explainers to generate local explanations (word and feature imporances) for the individual prediction.

        Input Parameters:

        fasttext_model (<fasttext.FastText._FastText> model): a supervised, pre-trained FastText model for extracting 
                                                                  dense text embeddings, trained on training data from
                                                                  four domain-specific datasets
                                                    
        classifier_model (sklearn.calibration.CalibratedClassifierCV): the pre-trained Calibrated Classifier that is
                                                                    wrapping a Passive-Aggressive base classifier for 
                                                                    outputting probabilities

        fitted_scaler (sklearn.preprocessing._data.StandardScaler): a pre-trained scikit-learn StandardScaler, on the same 
                                                                 combined dataset, for scaling the extra features

        text (str): the text to get the fake or real news prediction for

        feature_extractor (BasicFeatureExtractor): a class instance for extracting extra semantic and linguistic features from a text

        num_features (int): the number of top word features LIME should output importance scores for

        num_perturbed_samples (int): the default or user-adjusted number of perturbed samples to use for LIME explanations

        Output:
            dict: stores a summary of the information for explaining the prediction using LIME. Contains

                - "explanation_object": the default LIME explanation object returned by the LIME text explainer
                - "word_features_list": the list of tuples containing words and their LIME importance scores
                - "extra_features_list": a list of tuples containing the extra engineered features and their importance scores
                - "highlighted_text": the original text string formatted with HTML markup tags for displaying color-coded word features
                - "probabilities": an array of probabilities for [real, fake] news
                - "main_predicition": the final prediction, meaning integer of 0 for real news, 1 for fake news
    """

    # Creates a Lime Text Explainer
    text_explainer = LimeTextExplainer(class_names=["real", "fake"]) # Requires a mapping for 0 = real, 1 =fake

    # Extracts the extra engineered semantic and linguistic features for this text into a single-row Dataframe
    single_text_df = feature_extractor.extractFeaturesForSingleText(text)

    # Adds the original text as a column to the single-line features DF
    single_text_df["text"] = text
    
    # Extracts and stores ONLY the extra features (without the original text) in a features only Dataframe
    extra_features = single_text_df.drop("text", axis=1)

    # Extracts (as a list) the extra feature column names from that single-row features DataFrame for later mappings
    extra_feature_names = extra_features.columns.tolist()

    # Extracts the fastText embedding for the user-inputted text, inc. preprocessing function
    fasttext_embedding = getFastTextEmbedding(fasttext_model, text)

    # Stores the np fastText embedding in the single-row Dataframe
    single_text_df["fasttext_embeddings"] = [fasttext_embedding]

    # Returns the scaled and combined feature vector for the original input text
    text_features = combineFeaturesForSingleTextDF(single_text_df, fitted_scaler, extra_feature_names)

    # Predicts the news class and the class probabilities using the pre-trained Calib wrapping Passive-Aggressive news classifier
    text_prediction = classifier_model.predict(text_features)[0] # Extracts the single pred., as classifier returns a 2D array

    # Extracts the probability array from the CalibratedClassifierCV's predict_proba function
    text_probability_array = classifier_model.predict_proba(text_features)[0] 

    
    def predict_fn(perturbed_texts):
        """
        An inner function which LIME uses in order to predict probabilities of randomly-changed, perturbed texts.
        Important Note: use an inner functoion as this is only to be integrated with the LIME process, and therefore it has to access
        many variables inside the explainPredictionWithLIME, such as classifier models, scalers etc. to generate predictions.
        
        Input Parameters:
            perturbed_texts (list): a list of randomly changed, perturbed texts generated by LIME, with random words removed
        
        Output:
            perturbed_probs (numpy.ndarray): the outputted array of probabilities for the inputted perturbed news texts
        """

        # Stores the perturbed texts' extra features counts/values here
        perturbed_text_features_df_list = []
        
        # Iterates over each of the LIME-perturbed texts
        for perturbed_text in perturbed_texts:
            # Extracts the features for each perturbation of original text (outputs single row DF for later concat)
            df = feature_extractor.extractFeaturesForSingleText(perturbed_text)
            # Adds the actual news text to the single-row DataFrame for this perturbed text
            df["text"] = perturbed_text
            # Adds the fastText embedding to the single-row DataFrame
            df["fasttext_embeddings"] = [getFastTextEmbedding(fasttext_model, perturbed_text)]
            # Ads the single-row DataFrame to the whole list of DataFrames for later concatenation
            perturbed_text_features_df_list.append(df)
        
        # Concatenates the single perturbed text rows into one whole DataFrame
        perturbations_df = pd.concat(perturbed_text_features_df_list, axis=0, ignore_index=True)
        
        # Extracts, scales and concatenates extra features with fastText embeddings into arrays for model inputs
        perturbed_feature_arrays = []

        # Iterates over rows in the combined, perturbed features and text DataFrame for the changed samples
        for i in range(len(perturbations_df)):
            # Extracts a single row of perturbed features and text
            perturbed_text_row = perturbations_df.iloc[[i]]
            # Combines and scales the extra features with the fastText embeddings, outputs a single array of features
            perturbed_features = combineFeaturesForSingleTextDF(perturbed_text_row, fitted_scaler, extra_feature_names)
            # Appends the single text's features to the list storing all perturbed features for later stacking
            perturbed_feature_arrays.append(perturbed_features)
        
        # Stacks the list of rows of features into a single numpy features matrix (2D array)
        perturbed_feature_arrays = np.vstack(perturbed_feature_arrays)
        
        # Uses the trained model to output the probabilities for all perturbed samples
        perturbed_probs = classifier_model.predict_proba(perturbed_feature_arrays)
        
        # Return probs of permuted samples to LIME explainer to generate explanations based on prob differences to original text
        return perturbed_probs

    # Generates the LIME explanation using .explain_instance()
    # Reference: https://lime-ml.readthedocs.io/en/latest/lime.html
    """"
        Docs: "First, we generate neighborhood data by randomly perturbing features from the instance [...]
        We then learn locally weighted linear models on this neighborhood data to explain each of the classes in an interpretable way (see lime_base.py)."
    """
    explanation = text_explainer.explain_instance(
        text, # Original text
        # Docs: a required "classifier prediction probability function, which takes a numpy array and outputs prediction
        #  probabilities. For ScikitClassifiers, this is classifier.predict_proba."" This is the inner func. above.
        predict_fn,
        num_features=num_features, # The number of top/most important words to output LIME importance scores for
        num_samples=num_perturbed_samples, # The number of perturbed versions of the text to generate; the more the greater accuracy, but takes more time
        labels=[text_prediction] # "Explains" the main original text prediction, so words pushing to main prediction get POSITIVE scores
    )
    
    # Returns the word feature explanations as a list of tuples of (word, importance_score)
    word_features = explanation.as_list(label=text_prediction)

    # Filters out important words which are stopwords for more meaningful explanations
    word_features_filtered = [(word_feature[0], word_feature[1]) for word_feature in word_features
                             if word_feature[0].lower() not in stop_words] 

    # Sorts the text_feature_list in descending order by absolute value of importance scores
    word_feature_list_sorted = sorted(word_features_filtered, key = lambda x: abs(x[1]), reverse=True)


    ## REPLACED BY LIME TABULAREXPLAINER TO GENERATE WORD AND SEMANTIC/LINGUISTIC FEATURE SCORES ON THE SAME SCALE
    # # Creates a list to store the extra features'importance in this list of tuples (feature_name, feature_importance)
    # extra_feature_importances= []

    # # Iterates over extra features
    # for feature in extra_feature_names:

    #     # Creates a perturbed version of the original single text's row, but with the feature's value zeroed out to eval its importance
    #     perturbed_df = single_text_df.copy()

    #     # Zeroes out the current feature
    #     perturbed_df[feature] = 0

    #     # Outputs the features array for the original text but with this feature zeroed out
    #     features_perturbed = combineFeaturesForSingleTextDF(perturbed_df, fitted_scaler, extra_feature_names)

    #     # Returns the [real, fake] probability array for the text with the feature zeroed out
    #     perturbed_probability_array = classifier_model.predict_proba(features_perturbed)[0]
        
    #     # Calculate the importance of the current feature to the main prediction:
    #     # This is done by calculating the difference between the main pred probability for original and perturbed text
    #     feature_importance = text_probability_array[text_prediction] - perturbed_probability_array[text_prediction]

    #     # Append a tuple storing name of the zeroed-out current feature and its importance based on probability difference
    #     extra_feature_importances.append((feature, feature_importance))

    # Extracts the extra feature scores as a np array
    extra_features_array = single_text_df[extra_feature_names].values

    # Creates a LIME tabular explainer for explaining the impact of the engineered features on the predictions
    """"
        "Explains predictions on tabular (i.e. matrix) data. For numerical features, 
        perturbs them by sampling from a Normal(0,1) and doing the inverse operation of mean-centering and scaling, 
        according to the means and stds in the training data" - from the **LIME TabularExplainer docs**
    """
    tabular_explainer = LimeTabularExplainer(
        # Uses the extra features array generated from the inputted sample to perturb their values
        training_data=extra_features_array,
        feature_names=extra_feature_names, # Sends in the names of the extra features
        class_names=["real", "fake"],
        discretize_continuous=False, # Does NOT group continuous features into quartile bins
        mode="regression",  # Uses the probability differences rather than binary labels 
        sample_around_instance=True # "Will sample continuous features in perturbed samples from a normal centered at the instance being explained."" - Docs
    )

    def extra_features_predict_fn(perturbed_featuresets):
        """"
            Generates probabilities using the pre-trained scaler and model for the extra features, for us
            with the LIME Tabular Explainer

                Input Parameters:
                    perturbed_featuresets (list or array-like): a list of the perturbed numerical extra feature scores sampled around the original news sample

                Output:
                    two-dimensional list/array: a list or array-like structure containing the probabilities for the perturbed samples for LIME Tab-Explainer
        """

        # Stores the scaled, extra perturbed featuresets with added, stacked FastText embeddings
        combined_featuresets = []

        # Iterates over the number of LIME-perturbed extra-engineered featuresets
        for i in range(perturbed_featuresets.shape[0]): 

            # Creates a temporary DataFrame for the current perturbed feature set
            current_df = pd.DataFrame([perturbed_featuresets[i]], columns=extra_feature_names)

            # Adds the FastText embedding to the current extra engineered featureset
            current_df["fasttext_embeddings"] = [fasttext_embedding]

            # Scales and combines the embeddings and the extra features
            combined_sample = combineFeaturesForSingleTextDF(current_df, fitted_scaler, extra_feature_names)

            # Appends to the list storing all combined, scaled f sets
            combined_featuresets.append(combined_sample)

        # Stacks the rows of combined feature sets into one 2D array of the shape num_perturbed_samples x num_all_features_plus_embedding_size
        combined_features_array = np.vstack(combined_featuresets)

        # Outputs the predicted probabilities for the perturbed samples
        return classifier_model.predict_proba(combined_features_array)
    
    # Generates the LIME tabular explanations for the extra engineered semantic and linguistic features
    extra_features_explanation = tabular_explainer.explain_instance(
        extra_features_array[0], # Extracts the single row from the original extra features array
        extra_features_predict_fn, # The inner function to use with the explainer
        num_features=len(extra_feature_names), # Number of xtra feats (10)
        labels=[text_prediction], # Output feature importance scores where positive scores are ones pushing towards this label
        num_samples=max(num_perturbed_samples, 100) # Caps the extra feature perturbations to 100, as there are only 10 features
    )

    # Extracts the extra feature importance scores
    extra_feature_importances = extra_features_explanation.as_list(label=text_prediction)
    
    # Sorts the extra features importance values by absolute importance in descending order
    extra_feature_importances.sort(key=lambda x: abs(x[1]), reverse=True)

    # Generates highlighted text using the func below --> it outputs a string with HTML-formatted text
    highlighted_text = highlightText(text, word_feature_list_sorted, text_prediction)
    
    # Returns the explanation dictionary with LIME explanations, sorted word + extra feature lists, highlighted text + main preds
    return {
        "explanation_object": explanation,
        "word_features_list": word_feature_list_sorted,
        "extra_features_list": extra_feature_importances,
        "highlighted_text": highlighted_text,
        "probabilities": text_probability_array,
        "main_prediction": text_prediction
    }




def highlightText(text, word_feature_list, text_prediction):
    """
    A function for highlighting the words in the input text based on their importance scores and class label the words are pushing
    the classifier towards using HTML <span> + inline CSS tags for color-coding for red=fake news, blue=real news.

    Input Parameters:
        text (str): the text to highlight
        word_feature_list (list of tuples): list of word-feature, importance-score tuples outputted by the LIME text explainer

    Output:
        str: HTML formatted string with tags designating the highlighted text
    """
    
    # Stores the dicts containing pos, colors and opacities for highlighting parts of the text with HTML tags based on word importance
    highlight_positions = []
    
    # Calculates the maximum absolute importance score from all the word features
    max_importance = max(abs(importance) for feature, importance in word_feature_list)

    # Iterates over important words to hihglight them and find their pos in the text
    for word_feature, importance in word_feature_list:
        
        # A placeholder for where to start searching for in the original text that will be updated during the loop
        pos = 0 # Starts at first character in the text

        # Breaks out of loop when no more of this word feature are found in the text
        while True:

            # Finds the first occurence of important word in the remaining text
            pos = text.lower().find(word_feature.lower(), pos)
            
            # If word is not found in remaining text, break out of while loop
            if pos == -1:
                break
                
            # Checks if word feature found is a valid word, not a sub-word of a different word
            # Returns its pos in the text (character indices) if it is a real word, None if it is not
            boundary_positions = detectWordBoundaries(text, pos, word_feature)

            # If word feature is just a subword, increment pos and move to next part of the text string, restart at beginning of while loop
            if boundary_positions is None:
                pos += 1 
                continue
            
            # If word features is a whole word, unpack its beginning and end pos (character indices)
            word_start_pos, word_end_pos = boundary_positions

            # Maps the colors to which class the feature is pushing the classifier towards
            # If the word feature importance is > 0, it means the feature is pushing it towards the main prediction
            if importance > 0:
                # If feature has same importance score as the main prediction, make sure real news = blue, and fake = red
                color = "blue" if text_prediction == 0 else "red" 
            else:
                # If feature has same importance score as the main prediction, make sure if main pred is real news,
                # then opposing word feature will be in red, if main pred is fake, opposing feature will be blue
                color = "red" if text_prediction == 0 else "blue"
            
            # Adds the dict storing word positions, color, opacity for highlighting the whole text
            highlight_positions.append({
                "start": word_start_pos, 
                "end": word_end_pos, 
                "color": color,
                "opacity": abs(importance) / max_importance if max_importance != 0 else 0, # Maps color alpha channel to abs feat importance
                "text": text[word_start_pos:word_end_pos]
            })
            
            # Moves past this word by incrementing the string index to be at the character index just after this word
            pos = word_end_pos  

    # Sorts the word position dictionaries in ascending order based on word start index, to go from beginning to end of text
    highlight_positions.sort(key = lambda x: x["start"])
    
    # Merges the adjacent highlights of the same color to represent bigrams and trigrams and continuous text sequences, if have same color
    merged_positions = []

    # Only if there is more than 1 dictionary in the highlight_positions list then proceed
    if highlight_positions:

        # Extracts the first dictionary (storing text segment information) for highlighting
        current = highlight_positions[0]

        # Iterates through all of the next highlighting information dictionaries after this current one
        for next_pos in highlight_positions[1:]:

            # Checks if the next segment to be highlighted either starts right after current one ends or overlaps (add 1 to account for spaces!!!)
            if (next_pos["start"] <= current["end"] + 1 and
                # Only proceeds if the colors of highlighted words are the same
                next_pos["color"] == current["color"]):

                # Merges the words by extending current end position to be the one LATER at the end of the next segment instead
                current["end"] = max(current["end"], next_pos["end"])

                # Uses whichever opacity/word importance was strongest between the two words when merging the section of continous text
                current["opacity"] = max(current["opacity"], next_pos["opacity"])

                # Selects the words to be highlighted based on the new positions
                current["text"] = text[current["start"]:current["end"]]

            else:
                # If current word cannot merge with the next important word because they don't overlap, just add the current word dict to the list
                if next_pos["start"] > current["end"]: 
                    merged_positions.append(current)
                    # Moves to the next wor importance dict
                    current = next_pos
                else:
                    # If words overlap but have different colors (blue for real and red for fake), skip the word with the weaker importance or opacity
                    if next_pos["opacity"] > current["opacity"]:
                        current = next_pos
        
        # Adds the the current word segment
        merged_positions.append(current)  

    # Stores the highlighted text containing HTML tags
    result = []

    # Uses this to track pos in the text to start highlighting
    last_end = 0
    
    # Iterates over the dicts specifying highlighting positions and opacities (including merged consecutive sections of words)
    for pos in merged_positions:

        # If the next segment-to-highlight's start pos is greater than last_end, this means the previous text is NOT highlighted
        # so just add that previous plain text to the final result list (that will be joined into a string) and proceed
        if pos["start"] > last_end:
            result.append(text[last_end:pos["start"]])
        
        # Maps the segment dict's color label to the appropriate RGB color to use in in-line CSS HTML tags 
        color = "rgba(255, 0, 0," if pos["color"] == "red" else "rgba(0, 90, 156,"  # red (fake) or dodger blue (news)

        # Sets the value for the alpha opacity channel
        background_color = f"{color}{pos['opacity']})"
        
        # Appends the section of highlighted text with a span HTML tag and the inline CSS styling to the resulting highlighted text
        result.append(
            f"<span style='background-color:{background_color}; font-weight: bold'>"
            f"{pos['text']}</span>"
        )
        
        # Updates the last_end tracker to the final index of the highlighted section, to shift to the next part of the text
        last_end = pos["end"]
    
    # After it is done iterating through all highlighted segments, add any remaining non-highlighted text after last_end to the result list
    if last_end < len(text):
        result.append(text[last_end:])

    # Rejoins the result list containing HTML markup for highlighting into a string
    return "".join(result)



def detectWordBoundaries(text, word_start_pos, word):
    """
    A function ensuring that the highlighting func. is only matching whole words, and not parts of 
    words (e.g. avoid highlighting the word feature "hand" when iterating over the word "handle")

    It returns the start and end position if it's a valid word boundary (start and ende of word), but returns None otherwise

    Input Parameters:
        text (str): the whole text the word to highlight is part of
        word_start_pos (int): starting position of word (feature)
        word (str): the word feature that should be highlighted

    Output:
        the tuple of word_start_pos (int), word_end_pos (int): return positions only if word is indeed a word surrounded by a boundary
        None: returns None if there is no valid word boundary around this instance of the word substring
    """

    # Defines the series of punctuation characters for detecting word boundaries, such as space, exclamation mark, hyphen, etc.
    boundary_chars = set(' .,!?;:()[]{}"\n\t-')
    
    # Checks for word boundary at start of the word to highlight: passes check if position is 0 (start of text) 
    # # or if previous character is in the boundary_chars
    start_check = word_start_pos == 0 or text[word_start_pos - 1] in boundary_chars

    # Calculates the word's end position by adding the start idx to length of word
    word_end_pos = word_start_pos + len(word)

    # Checks for word boundary at end of word to highlight by comparing end pos to either length of whole text | if last pos is boundary char
    end_check = word_end_pos == len(text) or text[word_end_pos] in boundary_chars

    # If both start_check and end_check are set to True, returns the word start and end positions
    if start_check and end_check:
        return word_start_pos, word_end_pos

    # If the word is part of a larger word, returns None instead of the positions
    return None


def createCombinedFeatureDataFrame(explanation_dict, extra_features_df):
    """
    A helper function for plotting combined features. It creates a combined DF for both top word and extra feats importance scores

    Input Parameters:
        explanation_dict (dict): a dictionary storing important features lists generated by LIME explainers and the main prediction proba array
        extra_features_array (pd.DataFrame): a pandas DataFrame storing importance score for the extra engineered sem/linguistic features

    Output:
        combined_text_and_extra_engineered_features_df (pd.DataFrame): a pandas DataFrame storing combined word and extra features
                                                                       sorted by absolute feature importance score
    """
        
    # Gets all the top 10 BOTH real & fake word features by absolute importance score
    all_top_word_feats = sorted(
        explanation_dict["word_features_list"], # Gets the dict storing LIME explanations for top word features
        key=lambda word_feature_tuple: abs(word_feature_tuple[1]),
        reverse=True # Gets the top word feats in desc order by importance score
    )

    # Extracts the top 10 word features from the sorted dict
    top_10_word_feats = all_top_word_feats[:10]

    # Converts the top words dictionary to a pandas DataFrame
    top_word_feats_df = pd.DataFrame(top_10_word_feats, columns=["Feature", "Importance"])
    # Adds a new column to specify which type of feature this is (word or extra engineered feature)
    top_word_feats_df["Feature Type"] = "Word"

    # Creates a copy of the extra engineered features DataFrame
    top_extra_feats_df = extra_features_df[["Feature", "Importance"]]
    # Adds a new column to indicate feature type
    top_extra_feats_df["Feature Type"] = "Semantic / Linguistic"

    # Wraps the word features in single quote marks, so users can distinguish between word features and extra engineered features
    top_word_feats_df["Feature"] = top_word_feats_df["Feature"].apply(lambda word: f"'{word}'")

    # Concatenates the two DataFrames storing top features, each feat in a row
    combined_text_and_extra_engineered_features_df = pd.concat([top_word_feats_df, top_extra_feats_df], ignore_index=True)

    # Adds absolute importance column to sort the combined features
    combined_text_and_extra_engineered_features_df["Absolute Importance"] = combined_text_and_extra_engineered_features_df["Importance"].abs()

    # Sorts the DataFrame by absolute importances scores
    combined_text_and_extra_engineered_features_df = combined_text_and_extra_engineered_features_df.sort_values("Absolute Importance", ascending=False)

    return combined_text_and_extra_engineered_features_df


def displayAnalysisResults(explanation_dict, container, news_text, feature_extractor, FEATURE_EXPLANATIONS):

    """
    Displays comprehensive LIME prediction analysis results including predivted label, prediction probabilities, 
    confidence scores, and feature importance charts.
    
    Input Parameters:

        explanation_dict (dict): dict containing LIME explanation results, including word and extra feature scores, 
                                 and HTML-marked up highlighted text
        
        container: the instance of the Streamlit app container to display the results in (passed in inside of app.py)
        
        news_text (str): the user's inputted text to generate prediction and LIME explanations for
        
        feature_extractor (an instance of the BasicFeatureExtractor class): for processing the inputted
                                                                            text to get semantic and linguistic features
        
        FEATURE_EXPLANATIONS (dict): natural language explanations of the different exra engineered non-word semantic
                                     and linguistic features
    """
    
    # Converts the news category label from 0 or 1 to text labels
    main_prediction = explanation_dict["main_prediction"]
    main_prediction_as_text = "Fake News" if explanation_dict["main_prediction"] == 1 else "Real News"
    
    # Extracts the [real, fake] probability array returned from LIME explainer function
    probs = explanation_dict["probabilities"]
    
    # Displays the prediction results based on LIME explainer output
    container.subheader("Text Analysis Results")
    # Writes out the general predicted label
    container.markdown(f"**General Prediction:** {main_prediction_as_text}")
    # Write out the probabilities of being real and fake news
    container.markdown("**Confidence Scores:**")

    # Sets the color and boldness with markdown and HTML depending on the predicted label
    if main_prediction_as_text == "Real News":
        container.markdown(f"- <span style='color:dodgerblue; font-weight: bold'>Real News: {probs[0]:.2%}</span>", unsafe_allow_html=True)
        container.markdown(f"- <span style='color:red'>Fake News: {probs[1]:.2%}</span>", unsafe_allow_html=True)
    else:
        container.markdown(f"- <span style='color:dodgerblue'>Real News: {probs[0]:.2%}</span>", unsafe_allow_html=True)
        container.markdown(f"- <span style='color:red; font-weight: bold'>Fake News: {probs[1]:.2%}</span>", unsafe_allow_html=True)


    
    # Displays the highlighted text title using Markdown and inline CSS styling to adjust the size and padding
    st.markdown("""
        <div style='padding-top: 20px; padding-bottom:10px; font-size: 24px; font-weight: bold;'>
            Highlighted Text Section
        </div>
        """, unsafe_allow_html=True  # Reference: https://discuss.streamlit.io/t/unsafe-allow-html-in-code-block/54093
    )
    
    # Displays the explanation for what the color coding means in the highlighted text
    st.markdown("""
        <div style='padding-bottom:12px; padding-top: 12px; font-size: 18px; font-style: italic;'>
        Highlighted text shows the words (features) pushing the prediction towards <span style="color: dodgerblue; font-weight: bold;">real news</span> in blue 
        and to <span style="color: red; font-weight: bold;">fake news</span> in red.
        </div>
    """, unsafe_allow_html=True)
    
   
    # "Insert a multi-element container that can be expanded/collapsed."
    # Reference: https://docs.streamlit.io/develop/api-reference/layout/st.expander
    with st.expander("View Highlighted Text:"):
        # Formats the expandable scroll-box to show highlighted (blue=real, red=fake) text outputted by LIME Explainer using
        # inline CSS to allow y-scrolling and padding
        st.markdown("""
            <div style='height: 450px; overflow-y: scroll; border: 2px solid grey; padding: 12px;'>
                {}
            </div>  
            """.format(explanation_dict["highlighted_text"]), unsafe_allow_html=True)
        
    # Adds more padding between the two charts
    st.markdown("""
                    <style>
                            .stColumn > div {
                                padding: 0 20px;
                            }
                    </style>
    """, unsafe_allow_html=True)
    
    # Adds the title for bar charts for feature importance analysis
    container.subheader("Feature Importance Analysis")

    # Creates two columns for side-by-side bar charts for word features on the left and extra features on the right
    col1, col2 = container.columns(2)
    
    # First column: word features importance scores
    with col1:
        col1.markdown("### Top Word Features")

        # Extracts the top text-based features identified by LIME into a DataFrame for easier sorting and filtering
        word_features_df = pd.DataFrame(
            explanation_dict["word_features_list"],
            columns=["Feature", "Importance"]
        )

        # Filters DataFrame to select ONLY the words that pushed the classifier towards the main prediction (positive word scores)
        main_prediction_filtered_word_features_df = word_features_df[word_features_df["Importance"] > 0].copy()

        # Filters DataFrame to select ONLY words pushing towards the opposite class (negative word scores)
        opposite_prediction_filtered_word_features_df = word_features_df[word_features_df["Importance"] < 0].copy()

        # Sets the main prediction words graphs' titles based on whether main prediction is real or fake news
        if main_prediction == 0:  # Real news
            title = "Words pushing towards REAL NEWS"
        else: # Main prediction is fake news
            title = "Words pushing towards FAKE NEWS"
        
        # Sets the opposing prediction words graphs' titles based on whether main prediction is real or fake news
        if main_prediction == 0:  # Real news
            opposite_title = "Words pushing towards FAKE NEWS"
        else: # Main prediction is fake news
            opposite_title = "Words pushing towards REAL NEWS"

        # Extracts the top ten features of the words pushing towards the main prediction filtered DataFrame
        main_prediction_top_word_features_df =  main_prediction_filtered_word_features_df.nlargest(10, "Importance")

        # Extracts the top ten features of the words pushing towards the opposing prediction filtered DataFrame
        opposite_prediction_top_word_features_df = opposite_prediction_filtered_word_features_df.nlargest(10, "Importance")

        # Calculate the maximum importance value for pro-prediction word features and opposing-prediction word features
        # This is very important for scaling the y-axis on the two charts
        max_importance_score = 0

        # Calculates the max important score from the word features pushing towards the main prediction
        if len(main_prediction_top_word_features_df) > 0:
            max_importance_score = max(max_importance_score, main_prediction_top_word_features_df["Importance"].max()) # Updates max score if word importance max greater than 0

        # Calculates the ABSOLUTE max importance score from the opposing word feature scores (as they are negative)
        if len(opposite_prediction_filtered_word_features_df) > 0:
            opposite_max = opposite_prediction_filtered_word_features_df["Importance"].abs().max()
            # Update the max score if a greater value is found from the opposing word features
            max_importance_score = max(max_importance_score, opposite_max)

        # Adds a small offset to the top of the chart to make it look neater
        max_importance_score = max_importance_score * 1.1

        # Checks that there ARE important word features in the main prediction DataFrame
        if len(main_prediction_top_word_features_df) > 0:
            # Creates a bar chart for most important text features using the Altair visualization library
            # Q = specifies this feature/value is quantitative, N = specifies it is nominal/categorical
            # Reference: https://altair-viz.github.io/user_guide/generated/toplevel/altair.Chart.html#altair.Chart
            # Reference for sorting documentation: https://altair-viz.github.io/user_guide/generated/core/altair.EncodingSortField.html
            # How to create charts in Streamlit Reference: https://www.projectpro.io/recipes/display-charts-altair-library-streamlit
            # Use mark_bar to create bar_chart. Reference: https://altair-viz.github.io/user_guide/marks/index.html
            word_features_chart = alt.Chart(main_prediction_top_word_features_df).mark_bar().encode( 
                # Displays the categorical features (:N)/words on the x-axis
                x=alt.X(
                    "Feature:N", # N = categorical variable
                    sort=alt.EncodingSortField(
                        field="Importance",  # Sorts by word importance (for main pred, importance goes from pos value to 0)
                        order="descending" 
                    ),
                    title="Word Feature",
                    axis=alt.Axis(
                        labelAngle=-45,  # Uses a rotation of 45 degrees for users to be able to read labels better
                        labelLimit=150,  # Maximum allowed pixel width of axis tick labels. Reference: https://altair-viz.github.io/user_guide/generated/core/altair.Axis.html
                        labelOverlap=False  # Prevents the labels from overlappimg
                    )),
                # Plots the importance scores on y-axis 
                y=alt.Y( 
                        "Importance:Q", # Q means this is a quant. numerical value for Altair
                        title="Importance Strength",
                        # Reference: https://altair-viz.github.io/user_guide/generated/core/altair.Scale.html
                        scale=alt.Scale(domain=[0, max_importance_score]) # Set fixed scale for easier comparison
                        ),
                # Sets blue bars for real news prediction, red for fake 
                color=alt.value("dodgerblue") if main_prediction == 0 else alt.value("red"), 
                # Adds "tooltips": explanations that appear when hovering above the bar
                tooltip=["Feature",
                        alt.Tooltip("Importance", title="Word Importance")]
            ).properties(
                title=title, # Sets the title and chart dimensions
                height = 600,
                width=500
            ).configure_axis( # Reference: https://altair-viz.github.io/altair-viz-v4/user_guide/configuration.html
                labelFontSize=14,
                titleFontSize=16
            )

            # Displays the word features pushing towards the main prediction chart
            col1.altair_chart(word_features_chart , use_container_width=True)
        else:
            # If no important words have been found for pushing towards the main prediction, then print error message
            st.warning("No significant word features pushing the classifier towards the main prediction have been found.")


        # Creates the word features chart for words pushing towards the opposite (lower prob) prediction
        if len(opposite_prediction_top_word_features_df) > 0:
            
            # Modifies the original DF to get absolute importance scores for easier plotting in desc order from 0 to high
            opposite_df_for_chart = opposite_prediction_top_word_features_df.copy()

            # Applies abs func to whoel Importance col
            opposite_df_for_chart["Importance"] = opposite_df_for_chart["Importance"].abs()

            # Sorts the (now absolute) importance values in desc order
            opposite_df_for_chart = opposite_df_for_chart.sort_values("Importance", ascending=False)

            # Creates the chart with the sorted abs values pushing towards the opposite prediction
            opposite_word_features_chart = alt.Chart(opposite_df_for_chart).mark_bar().encode(
                x=alt.X(
                    "Feature:N", # x-axis = categorical value (word feature)
                    sort=alt.EncodingSortField(
                        field="Importance",
                        order="descending"
                    ),
                    title="Word Feature",
                    axis=alt.Axis(
                        labelAngle=-45, # Rotates labels for readability
                        labelLimit=150, # Set label length to 150 pixels
                        labelOverlap=False # Don't let labels overlap
                    )),
                y=alt.Y( # y-axis = word importance score (quantitative, number)
                    "Importance:Q",
                    title="Importance Strength",
                    scale=alt.Scale(domain=[0, max_importance_score])
                    ),
                # Uses the opposite color to the main prediction
                color=alt.value("red") if main_prediction == 0 else alt.value("dodgerblue"),
                tooltip=["Feature",
                        alt.Tooltip("Importance", title="Word Importance")]
            ).properties(
                title=opposite_title, # Adds title and dimensions
                height = 600,
                width=500
            ).configure_axis(
                labelFontSize=14, # Set thes label and title font szzies
                titleFontSize=16
            )
        
            # Displays the opposite word features chart
            col1.altair_chart(opposite_word_features_chart, use_container_width=True)

        else:

            # Prints an error message if no significant words pushing towards the opposing prediction are found
            st.warning("No significant word features pushing the classifier away from the main prediction have been found.")
        
    # The second column is for displaying a bar chart with the importance scores for non-word features
    with col2:
        
        # Extracts raw feature scores for the input news text to show users more info about what the actual value was
        raw_feat_scores = feature_extractor.extractRawFeaturesForSingleText(news_text)

        # Adds the title to the extra features chart
        col2.markdown("### Top Extra Semantic and Linguistic Features")
        
        # Creates a DataFrame from the extra features list returned by the LIME explainer function
        extra_features_df = pd.DataFrame(
            explanation_dict["extra_features_list"], # Extracts the extra feat importance scores from the LIME explainer dict
            columns=["Feature", "Importance"] # Creates a new DF with these col names
        )
        
        # Sorts the features by their absolute importance for easier plotting
        extra_features_df["Absolute Importance"] = extra_features_df["Importance"].abs()

        # Extracts the 10 largest features (all of the features) by absolute importance
        extra_features_df = extra_features_df.nlargest(10, "Absolute Importance")
        
        # Adds the original feature name, not the underscored programmatic title, before mapping to the explanation labels on the chart
        extra_features_df["Original Feature"] = extra_features_df["Feature"]
        
        # Maps the feature column variables to user-readable strings 
        feature_name_mapping = {
            "exclamation_point_frequency": "Exclamation Point Usage",
            "third_person_pronoun_frequency": "3rd Per. Pronoun Usage",
            "noun_to_verb_ratio": "Noun/Verb Ratio",
            "cardinal_named_entity_frequency": "Number Usage",
            "person_named_entity_frequency": "Person Name Usage",
            "nrc_positive_emotion_score": "Positive Emotion",
            "nrc_trust_emotion_score": "Trust Score",
            "flesch_kincaid_readability_score": "Readability Grade",
            "difficult_words_readability_score": "Difficult Words",
            "capital_letter_frequency": "Capital Letter Usage"
        }
        
        # Maps the features to their more readable names listed in the dictionary above
        extra_features_df["Feature"] = extra_features_df["Feature"].map(feature_name_mapping)
        
        # Maps the explanations to their globally-stored natural language explanation for users
        extra_features_df["Explanation"] = extra_features_df["Original Feature"].map(FEATURE_EXPLANATIONS)

        # Adds a new DF column to extra_features_df storing the raw feature scores for adding to the chart's tooltips to improve explanations
        extra_features_df["Raw Score"] = 0.0 # Define as all vals 0 to start
        # Iterate over the extra_features_df rows, 1 row per feat
        for index, row in extra_features_df.iterrows():
            # Extracts the raw feat score for each feature/row in extra_features_df
            # 1. Accesses the string defining the raw_feat column name using row["Original Feature"] from extra_features_df row
            # 2. Extracts the first and only row with idx 0 using .iloc
            raw_score = raw_feat_scores[row["Original Feature"]].iloc[0]
            # Updates the extra feature score to the extra_features_df for the chart's Tooltip  
            extra_features_df.at[index, "Raw Score"] = raw_score
        
        # Creates an Altair color-coded bar chart with more tooltips for explaining what features mean to users when they hover over a bar
        extra_features_chart = alt.Chart(extra_features_df).mark_bar().encode(
            x=alt.X("Importance:Q", title="Importance Strength"),
            y=alt.Y("Feature:N",
                    sort=alt.EncodingSortField(
                        field="Absolute Importance", # Sorts features by ABSOLUTE value of importance in desc order
                        order="descending"
                    ),
                    axis=alt.Axis(
                        labelFontSize=14 # Sets the feature labels font size
                    )), 
                color=alt.condition(
                # Checks if the importance score pushes in the same direction as the predicted class
                alt.datum.Importance > 0,
                alt.value("dodgerblue") if main_prediction == 0 else alt.value("red"), # If same dir as pred class
                alt.value("red") if main_prediction == 0 else alt.value("dodgerblue") # If NOT same dir as pred class
            ),
            tooltip=["Feature", # Adds hoverable tooltips showing the feature, importance score, and raw count/score if user hovers over the bar 
                    alt.Tooltip("Importance", title="Importance Score"), # Importance score
                    alt.Tooltip("Raw Score", title="Raw Feature Value", format=".4f") # Raw scores to 4 decimal places
                ]  
        ).properties(
            title="Top 10 Extra Features", # Sets title and dimensions
            height=600,
            width=500
        )
        
        # Displays the extra features chart in the second column
        col2.altair_chart(extra_features_chart, use_container_width=True) # Make span across whole container

        # Creates a new combined bar chart for comparing word and extra features
        col2.markdown("### Combined Top Word and Extra Features for Comparison")

        # Creates a combined DF for both top word and extra feats importance scores
        combined_text_and_extra_engineered_features_df = createCombinedFeatureDataFrame(explanation_dict, extra_features_df)

        # Creates the combined features chart
        combined_feats_chart = alt.Chart(combined_text_and_extra_engineered_features_df).mark_bar().encode(
            x = alt.X("Importance:Q", title="Importance Score"),
            y = alt.Y("Feature:N", 
                        sort=alt.EncodingSortField(
                            field="Absolute Importance",
                            order="descending"
                        ),
                        axis=alt.Axis(
                            labelFontSize=14,
                            labelLimit=200
                        )),
            color=alt.condition(
                # Checks if importance feature score is pos --> if it is means feature is pushing towards the main prediction
                alt.datum.Importance > 0,
                # If feature score is pos, and main pred. is real news (0), make bar blue (pushing towards real news), else if pred=fake, make it red
                alt.value("dodgerblue") if main_prediction == 0 else alt.value("red"),
                # If feature score is neg, make color opposite to that of the main prediction
                alt.value("red") if main_prediction == 0 else alt.value("dodgerblue")
            ),
            tooltip=[
                # Creates a hoverable tooltips to show importance score and kind of feature the bar represents
                "Feature",
                alt.Tooltip("Importance", title="Importance Score"),
                alt.Tooltip("Feature Type", title="Feature Type")
            ]
        ).properties(
            title="Combined Word + Extra Top Features",
            height = 600,
            width = 500
        )

        col2.altair_chart(combined_feats_chart, use_container_width=True)
    
    container.markdown("---")

    # Adds a legend explanation for the color-coded bar charts and highlighted text
    container.markdown(f"""
        **Legend:**
        - 🔵 **Blue bars**: Features pushing towards real news classification
        - 🔴 **Red bars**: Features pushing towards fake news classification
        
        - The length of each bar and color represent how strongly this feature
        in the news text pushed the classifier towards a REAL or FAKE prediction,
        not the raw feature score.
        
        *For more details about the extra semantic and linguistic features, including their raw scores and what they mean,
        please click below to view the detailed explanations.*
    """)
    
    # Adds an expander outlining the feature importance scores in more detail
    with container.expander("*View More Detailed Feature Score Information*"):
        
        # Iterates over the extra features to explain each one
        for index, row in extra_features_df.iterrows():

            # Determines the importance color and explanation based on the prediction and score value
            if row["Importance"] > 0: # Same direction of word importance as main prediction
                if main_prediction == 0: # Main prediction AND the importance score is real news
                    importance_color = "dodgerblue"
                    importance_explanation = "(pushing towards real news)"
                else:
                    importance_color = "red" # Main prediction AND the importancescore are fake news
                    importance_explanation =  "(pushing towards fake news)"    
            elif row["Importance"] < 0: # Different direction of word importance as main prediction
                if main_prediction == 0: # Main prediction is real BUT the word pushes towards fake news
                    importance_color = "red"
                    importance_explanation = "(pushing towards fake news)"
                else: # Main prediction is fake BUT the word pushes towards real news
                    importance_color = "dodgerblue"
                    importance_explanation =  "(pushing towards real news)"    
            else:
                importance_color = "grey"
                importance_explanation =  "Neutral - there was no significant impact on prediction."  

            raw_feat_score = raw_feat_scores[row["Original Feature"]].iloc[0]
            
            # Adds the explanations from FEATURE_EXPLANATIONS about the feature and main patterns associated with it in the training data
            container.markdown(f"""
            **{row["Feature"]}**  
            - **Raw Feature Score**: {raw_feat_score:.4f}  
            - **Feature Importance to Prediction**: <span style='color:{importance_color}'>{row["Importance"]:.4f} {importance_explanation}</span>  
            - **Explanation**: {FEATURE_EXPLANATIONS[row["Original Feature"]]}
            """, unsafe_allow_html=True)
