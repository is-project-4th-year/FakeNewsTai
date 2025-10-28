# Fourth or Final Iteration of Fake News Detection App

Contains the following new components:

1. Combined Top Word and Extra Features for Comparison chart, created by using LIME Tabular Explainer together with the LIME Text Explainer to output word features and extra features on the same scale, in response to user feedback. Word features have single quotes around them for clarity
2. Made the UI wider and thus less cramped in response to user feedback
3. Adjusted the widths and layout of the training data bar charts and WordClouds to be more readable and not too large
4. Made slight adjustments to the feature explanations, to make them clearer and highlight the difference between raw feature scores and importance scores
5. Created more robust URL scraping functions in app_utils.py to use beautifulsoup4 for scraping URLS in case newspaper3k does not work, as well as checking if the inputted text is in English, outputting an error message if it is in a different language


