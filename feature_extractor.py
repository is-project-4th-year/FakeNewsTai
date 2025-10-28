# Extracts semantic and linguistic features from a news text

# Imports text and data preprocessing libraries
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
nltk.download("punkt") # For tokenizer
nltk.download("punkt_tab")
# Imports libraries for feature extraction
import spacy
from nrclex import NRCLex
import textstat


class BasicFeatureExtractor:
    """
        A class containing methods for extracting key discriminative features for 
        helping an ML-based classifier categorize real and fake news. Inc. lexical features such as
        normalized (by text length in word tokens) exclamation point count, as well as semantic
        features (e.g. pos emotion score)
    """
    
    def __init__(self):
        # Loads in the SpaCy model for POS-tag + NER extraction
        self.nlp = spacy.load("spacy_model")


    def extractExclamationPointFreqs(self, text):
        """
        Extracts the frequencies of exclamation points from a single news text. 
        
            Input Parameters:
                text (str): the news text to extract exclamation point frequencies from
    
            Output:
                excl_point_freq (float): the normalized exclamation point frequency for the text.
                Normalized by num of word tokens to handle varying text length datasets
        """
        # Counts the number of exclamation points in the text
        exclamation_count = text.count('!')
        # Tokenizes text for calculating text length
        word_tokens = word_tokenize(text)
        # Calculates the text length in number of word tokens
        text_length = len(word_tokens)
        # Normalizes the exclamation point frequency by text length in tokens
        return exclamation_count / text_length if text_length > 0 else 0 # Handles division-by-zero errs
    

    def extractRawExclamationPointFreqs(self, text):
        """
        Extracts the RAW (non-normalized) frequencies of exclamation points from a single news text. 
        
            Input Parameters:
                text (str): the news text to extract exclamation point frequencies from
    
            Output:
                excl_point_freq (float): the raw exclamation point frequency for the text.
                NOT been normalized by num of word tokens to handle varying text length datasets
        """
        # Counts the number of exclamation points in the text and returns it
        return text.count('!')
    

    def extractThirdPersonPronounFreqs(self, text):
        """
        Extracts the normalized frequency counts of third-person pronouns in the inputted news text.
        
            Input Parameters:
                text (str): the news text to extract pronoun features from.
            
            Output:
                float: Normalized third-person pronoun frequency.
        """
        # Creates a alphab-ordered list of English 3rd person pronouns
        third_person_pronouns = [
            "he","he'd", "he's", "him", "his",
            "her", "hers", 
            "it", "it's", "its",
            "one", "one's", 
            "their", "theirs", "them","they", "they'd", "they'll", "they're", "they've",   
            "she", "she'd", "she's"
        ]

        # Tokenizes text for calculating text length
        word_tokens = word_tokenize(text)

        # Gets the text length in num tokens
        text_length = len(word_tokens)

        # Counts the frequency of third-person pronouns in the news text; lowercases text to match the list of third-person pronouns above
        third_person_count = sum(1 for token in word_tokens if token.lower() in third_person_pronouns)

        # Normalizes the frequency by text length in word tokens
        return third_person_count / text_length if text_length > 0 else 0
    
    
    def extractRawThirdPersonPronounFreqs(self, text):
        """
        Extracts the RAW frequency counts of third-person pronouns in the inputted news text.
        
            Input Parameters:
                text (str): the news text to extract pronoun features from.
            
            Output:
                float: Raw third-person pronoun frequency.
        """
        # Creates a alphab-ordered list of English 3rd person pronouns
        third_person_pronouns = [
            "he","he'd", "he's", "him", "his",
            "her", "hers", 
            "it", "it's", "its",
            "one", "one's", 
            "their", "theirs", "them","they", "they'd", "they'll", "they're", "they've",   
            "she", "she'd", "she's"
        ]

        # Tokenizes the text for calculating text length in word tokens
        word_tokens = word_tokenize(text)

        # Counts the frequency of third-person pronouns in the news text; lowercases text to match the list of third-person pronouns above
        third_person_count = sum(1 for token in word_tokens if token.lower() in third_person_pronouns)

        # Returns the frequency count
        return third_person_count



    def extractNounToVerbRatios(self, text):
        """
        Calculates the ratio of all types of nouns to all types of verbs in the text
        using the Penn Treebank POS Tagset and the SpaCy library with the downloaded
        "en_core_web_lg" model.
        
            Input Parameters:
                text (str): the news text to extract noun-verb ratio features from.
            
            Output:
                float: Noun-to-verb ratio, or 0.0 if there are 0 verbs in text
        """
        
        # Converts the text to an NLP doc object using SpaCy
        doc = self.nlp(text)
        
        # Defines the Penn Treebank POS tag categories for nouns and verbs
        # Reference: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
        noun_tags = ["NN", "NNS", "NNP", "NNPS"]
        verb_tags = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
        
        # Counts the freqs of both nouns and verbs based on the above Penn Treebank tags
        noun_count = sum(1 for token in doc if token.tag_ in noun_tags)
        verb_count = sum(1 for token in doc if token.tag_ in verb_tags)
        
        # Calculates and returns the noun-to-verb ratio (should be higher for fake news, as it had more nouns in EDA)
        return noun_count / verb_count if verb_count > 0 else 0.0 # Avoid division-by-zero error


    def extractCARDINALNamedEntityFreqs(self, text):
        """
        Extracts the normalized frequency of CARDINAL named entities in the text
        using the SpaCy library.
        
            Input Parameters:
                text (str): The text to extract the CARDINAL named entity frequencies from.
            
            Output:
                float: Normalized frequency (by number of tokens in the text) of CARDINAL named entities.
        """

        # Processes the text again with SpaCy to get NLP doc object
        doc = self.nlp(text)

         # Counts how many named entities have the label "CARDINAL"
        cardinal_entity_count = sum(1 for entity in doc.ents if entity.label_ == "CARDINAL")

        # Tokenizes the text
        word_tokens = [token for token in doc]
        
        # Counts the number of word toks
        text_length = len(word_tokens)

        # Returns the normalized frequency of CARDIAL named entities by word tok length
        return cardinal_entity_count / text_length if text_length > 0 else 0.0 # Avoid division-by-zero error
    

    def extractRawCARDINALNamedEntityFreqs(self, text):
        """
        Extracts the RAW frequency of CARDINAL named entities in the text
        using the SpaCy library.
        
            Input Parameters:
                text (str): The text to extract the CARDINAL named entity frequencies from.
            
            Output:
                float: RAW frequency (by number of tokens in the text) of CARDINAL named entities.
        """

        # Processes the text again with SpaCy to get NLP doc object
        doc = self.nlp(text)

        # Counts how many named entities have the label "CARDINAL"
        cardinal_entity_count = sum(1 for entity in doc.ents if entity.label_ == "CARDINAL")

        # Returns the raw named entity count
        return cardinal_entity_count


    def extractPERSONNamedEntityFreqs(self, text):
        """
        Extracts the normalized frequency of PERSON named entities in the text
        using the SpaCy library.
        
            Input Parameters:
                text (str): The text to extract the PERSON named entity frequencies from.
            
            Output:
                float: Normalized frequency (by number of tokens in the text) of PERSON named entities.
        """
        # Processes the text with SpaCy to get NLP doc object
        doc = self.nlp(text)
        
        # Counts how many named entities have the label "PERSON"
        person_entity_count = sum(1 for entity in doc.ents if entity.label_ == "PERSON")
        
        # Tokenizes the text
        word_tokens = [token for token in doc]
        
        # Counts num of word tokens
        text_length = len(word_tokens)
        
        # Returns the normalized frequency of PERSON named entities, normalized by dividing by text length in word tokens
        return person_entity_count / text_length if text_length > 0 else 0.0 # Avoids division-by-zero error
    

    def extractRawPERSONNamedEntityFreqs(self, text):
        """
        Extracts the RAW frequency of PERSON named entities in the text
        using the SpaCy library.
        
            Input Parameters:
                text (str): The text to extract the PERSON named entity frequencies from.
            
            Output:
                float: RAW frequency (by number of tokens in the text) of PERSON named entities.
        """
        # Processes the text with SpaCy to get NLP doc object
        doc = self.nlp(text)
        
        # Counts how many named entities have the label "PERSON"
        person_entity_count = sum(1 for entity in doc.ents if entity.label_ == "PERSON")
        
        # Returns the raw named entity count
        return person_entity_count


    def extractPositiveNRCLexiconEmotionScore(self, text):
        """
        Extracts the POSITIVE emotion score using the NRC Lexicon from the inputted news text.
        
            Input Parameters:
                text (str): the news text to extract POSITIVE emotion score from.
            
            Output:
                float: the POSITIVE words NRC Lexiconemotion score.
        """
        # Converts the text to lowercase to find (uncased) words in the lexicon
        text = text.lower()

        # Creates an NRC Emotion Lexicon object to extract emotion word freqs from
        emotion_obj = NRCLex(text)
        
        # Returns the POSITIVE emotion score (use "get" to default to 0.0 if for some reason it is not found)
        return emotion_obj.affect_frequencies.get("positive", 0.0) 


    def extractTrustNRCLexiconEmotionScore(self, text):
        """
        Extracts the TRUST emotion score using the NRC Lexicon from the inputted news text.
        
            Input Parameters:
                text (str): the news text to extract TRUST emotion score from.
            
            Output:
                float: the extracted TRUST emotion NRC lexicon score.
        """
        # Converts the text to lowercase to find (uncased) words in the lexicon
        text = text.lower()

        # Creates an NRC Emotion Lexicon object to extract emotion word freqs from
        emotion_obj = NRCLex(text)
        
        # Returns the TRUST emotion score (use "get" to default to 0.0 if for some reason it is not found)
        return emotion_obj.affect_frequencies.get("trust", 0.0)


    def extractFleschKincaidGradeLevel(self, text):
        """
        Extracts the Flesch-Kincaid Grade Level score for the input text.
        
        Input Parameters:
            text (str): the news text to calculate the Flesch-Kincaid Grade Level score.
        
        Output:
            float: the Flesch-Kincaid Grade Level score for the text.
        """
        return textstat.flesch_kincaid_grade(text)
        
    def extractDifficultWordsScore(self, text):
        """
        Extracts the number of difficult words (not in the Dall-Chall word list) in the input text using the textstat library.
        Reference about the Dall-Chall word list used to compute these scores: 
            - https://readabilityformulas.com/word-lists/the-dale-chall-word-list-for-readability-formulas/
        
        Input Parameters:
            text (str): the news text to calculate the difficult words score.
        
        Output:
            float: the number of difficult words score for the text.
        """
        # Converts text to lowercase to match if words are in difficult words list
        text = text.lower()
        
        # Returns the difficult words textstat score; the higher the score, the more non-Dall Chall words, the more coplex the text
        return textstat.difficult_words(text)
    

    def extractCapitalLetterFreqs(self, text):
        """
        Extracts the normalized frequency of capital letters in the input text.
        Normalized by the total number of word tokens.
    
            Input Parameters:
                text (str): The news text to extract capital letter frequencies from.
            
            Output:
                float: Normalized frequency of capital letters in the text.
        """
        # Counts the number of capital letters in the text
        capital_count = sum(1 for char in text if char.isupper())
        
        # Tokenizes the text
        word_tokens = word_tokenize(text)

        # Counts the number of word tokens in the text
        text_length = len(word_tokens)
        
        # Normalizes the frequency of capital letters
        return capital_count / text_length if text_length > 0 else 0.0 # Avoids division-by-zero error
    
    def extractRawCapitalLetterFreqs(self, text):
        """
        Extracts the RAW frequency of capital letters in the input text.
        Normalized by the total number of word tokens.
    
            Input Parameters:
                text (str): The news text to extract capital letter frequencies from.
            
            Output:
                float: Raw frequency of capital letters in the text.
        """
        # Counts the number of capital letters in the text
        capital_count = sum(1 for char in text if char.isupper())
        
        # Returns the raw capital letter frequency
        return capital_count
    

    def extractFeaturesForSingleText(self, text):
        """
        Extracts the basic features for classifying an individual news text, to concatenate with the fastText embeddings.

            - It outputs a single-row DataFrame because this will be used to extract the features
            for each perturbation of the text created to test the impact of removing different features on final probabilities. 
            - Each single-row DataFrame for the perturbed texts will then be concatenated into a multi-row DataFrame, storing the
            features for all of the perturbed texts in a single table, to use as model inputs.
    
        Input Parameters:
            text (str): The text to extract features from.
    
        Output:
            pd.DataFrame: A single-row DataFrame storing the extracted basic feature values for this text, each feature per column
        """
        # Extracts the features and stores them in a dict
        feature_dict = {
            "exclamation_point_frequency": self.extractExclamationPointFreqs(text),
            "third_person_pronoun_frequency": self.extractThirdPersonPronounFreqs(text),
            "noun_to_verb_ratio": self.extractNounToVerbRatios(text),
            "cardinal_named_entity_frequency": self.extractCARDINALNamedEntityFreqs(text),
            "person_named_entity_frequency": self.extractPERSONNamedEntityFreqs(text),
            "nrc_positive_emotion_score": self.extractPositiveNRCLexiconEmotionScore(text),
            "nrc_trust_emotion_score": self.extractTrustNRCLexiconEmotionScore(text),
            "flesch_kincaid_readability_score": self.extractFleschKincaidGradeLevel(text),
            "difficult_words_readability_score": self.extractDifficultWordsScore(text),
            "capital_letter_frequency": self.extractCapitalLetterFreqs(text),
        }

        # Converts the dict above to a DataFrame (it will contain a single row, as this is just one text)
        feature_df = pd.DataFrame([feature_dict])

        # Returns the dict with a single row, columns containing the feature names
        return feature_df
    
    def extractRawFeaturesForSingleText(self, text):
        """
        Extracts the basic raw, unnormalized features for showing users raw semantic and linguistic feature scores. 

            - It outputs a single-row DataFrame of raw feature scores
            - Can be used to inform users not only of the features' importance score, but of what the raw counts/scores were
    
        Input Parameters:
            text (str): The text to extract features from.
    
        Output:
            pd.DataFrame: A single-row DataFrame storing the raw extracted basic feature values for this text, each feature per column
        """
        # Extracts the raw freq counts for certain features or normal scores for others and stores them in a dict
        feature_dict = {
            "exclamation_point_frequency": self.extractRawExclamationPointFreqs(text), # Does not normalize excl. point count by text len
            "third_person_pronoun_frequency": self.extractRawThirdPersonPronounFreqs(text),
            "noun_to_verb_ratio": self.extractNounToVerbRatios(text),
            "cardinal_named_entity_frequency": self.extractRawCARDINALNamedEntityFreqs(text),
            "person_named_entity_frequency": self.extractRawPERSONNamedEntityFreqs(text),
            "nrc_positive_emotion_score": self.extractPositiveNRCLexiconEmotionScore(text),
            "nrc_trust_emotion_score": self.extractTrustNRCLexiconEmotionScore(text),
            "flesch_kincaid_readability_score": self.extractFleschKincaidGradeLevel(text),
            "difficult_words_readability_score": self.extractDifficultWordsScore(text),
            "capital_letter_frequency": self.extractRawCapitalLetterFreqs(text),
        }

        # Converts the dictionary above to a single-row DataFrame for consistency with previous approach
        feature_df = pd.DataFrame([feature_dict])

        # Returns the dict with a single row, columns containing the feature names
        return feature_df