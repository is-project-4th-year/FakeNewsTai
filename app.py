# app.py – TAIEYE Fake News Detector with Firebase Auth
import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os
import matplotlib.pyplot as plt
import joblib
import fasttext
import gdown
from feature_extractor import BasicFeatureExtractor
from app_utils import scrapeWithSoup, analyzeNewsText, detectIfTextIsEnglish
from newspaper import Config, Article

# ──────────────────────────────
# 1. Firebase & Auth Setup
# ──────────────────────────────
FIREBASE_CRED_PATH = "taieye-a98df-firebase-adminsdk-fbsvc-763c61ed82.json"

if not firebase_admin._apps:
    if os.path.exists(FIREBASE_CRED_PATH):
        firebase_admin.initialize_app(credentials.Certificate(FIREBASE_CRED_PATH))
    else:
        st.error(f"Missing `{FIREBASE_CRED_PATH}`. Add it to project root.")

CONFIG_FILE = "config.yaml"
if not os.path.exists(CONFIG_FILE):
    default = {
        'credentials': {'usernames': {}},
        'cookie': {'expiry_days': 30, 'key': 'taieye_secure_2025', 'name': 'taieye_auth'},
        'preauthorized': []
    }
    with open(CONFIG_FILE, 'w') as f:
        yaml.dump(default, f)

with open(CONFIG_FILE) as f:
    config = yaml.load(f, Loader=SafeLoader)

def create_user(email, password, name):
    try:
        user = auth.create_user(email=email, password=password)
        config['credentials']['usernames'][email] = {
            'email': email, 'name': name,
            'password': stauth.Hasher([password]).generate()[0]
        }
        with open(CONFIG_FILE, 'w') as f:
            yaml.dump(config, f)
        return True
    except Exception as e:
        st.error(f"Signup failed: {e}")
        return False

def send_magic_link(email):
    try:
        settings = auth.ActionCodeSettings(url="http://localhost:8501")
        link = auth.generate_sign_in_with_email_link(email, settings)
        st.code(link)
        st.info("Click the link above to sign in.")
    except Exception as e:
        st.error(f"Magic link failed: {e}")

def send_reset_link(email):
    try:
        auth.generate_password_reset_link(email)
        st.success("Reset link sent!")
    except Exception as e:
        st.error(f"Reset failed: {e}")

# ──────────────────────────────
# 2. Page Config + CSS + PERFECT LOGO + TITLE
# ──────────────────────────────
st.set_page_config(page_title="TAIEYE", page_icon="Eagle", layout="wide")

st.markdown("""
<style>
    .main {background: #0e1117; color: white; padding: 2rem 0;}
    .stButton>button {background: #ff4b4b; color: white; border-radius: 12px; font-weight: bold; padding: 10px 24px;}
    .logo-container {text-align: center; margin: 40px 0 30px;}
    .logo-img {width: 180px; border-radius: 50%; box-shadow: 0 4px 12px rgba(0,0,0,0.3);}
    .title {font-size: 3.2rem; font-weight: 700; margin: 20px 0 8px; color: #ff4b4b; letter-spacing: 1px;}
    .subtitle {font-size: 1.2rem; color: #aaa; margin-bottom: 30px;}
    .block-container {max-width: 2000px; padding: 0 2rem;}
    .eagle {width: 70px; animation: float 3s ease-in-out infinite; margin: 0 auto; display: block;}
    @keyframes float {0%,100%{transform:translateY(0)} 50%{transform:translateY(-15px)}}
</style>
""", unsafe_allow_html=True)

# ── LOGO + TITLE (BEFORE LOGIN) ──
st.markdown(f"""
<div class="logo-container">
    <img src="assets/images/taieye_logo.jpg" class="logo-img">
    <h1 class="title">TAIEYE</h1>
    <p class="subtitle"><i>Truth in Every Word</i></p>
    <img src="assets/images/taieye_logo.jpg" class="eagle">
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────
# 3. Authenticator
# ──────────────────────────────
authenticator = stauth.Authenticate(
    config['credentials'], config['cookie']['name'],
    config['cookie']['key'], config['cookie']['expiry_days']
)

# ──────────────────────────────
# 4. Login / Register / Magic Link
# ──────────────────────────────
tab1, tab2, tab3 = st.tabs(["Login", "Register", "Passwordless"])

with tab1:
    name, auth_status, username = authenticator.login('Login', 'main')
    if st.button("Sign in with Google"):
        st.info("Google OAuth: Use Email/Password for now.")
    if st.button("Send Magic Link"):
        email = st.text_input("Email for magic link", key="magic_email")
        if st.button("Send", key="send_magic"):
            send_magic_link(email)

with tab2:
    with st.form("register"):
        st.write("### Create Account")
        name_in = st.text_input("Full Name")
        email = st.text_input("Email")
        p1 = st.text_input("Password", type="password")
        p2 = st.text_input("Confirm", type="password")
        if st.form_submit_button("Register"):
            if p1 != p2:
                st.error("Passwords don't match")
            elif create_user(email, p1, name_in):
                st.success("Registered! Login now.")
                st.experimental_rerun()

with tab3:
    email = st.text_input("Email for Magic Link", key="magic_link_email")
    if st.button("Send Link", key="send_link"):
        send_magic_link(email)

with st.expander("Forgot Password?"):
    email = st.text_input("Your email", key="reset_email")
    if st.button("Send Reset Link", key="send_reset"):
        send_reset_link(email)

# ──────────────────────────────
# 5. AUTHENTICATED APP (Your Original Code)
# ──────────────────────────────
if auth_status:
    st.sidebar.success(f"Welcome, {name}!")
    authenticator.logout('Logout', 'sidebar')

    # ── YOUR ORIGINAL CODE STARTS HERE (unchanged) ──
    st.title("Fake News Detection App")

    FEATURE_EXPLANATIONS = {
        "exclamation_point_frequency": 
            """Exclamation mark frequency count. Higher raw ! counts in the training data
            signal more emotional or sensational writing, and were therefore more strongly associated 
            with fake news.""",
        
        "third_person_pronoun_frequency": 
            """Frequency of third-person pronouns (he, she, they, etc.). Higher raw counts indicate a more story-like 
            narrative style. The higher the original score score was in the training data, the more likely the text was to be fake news.""",
        
        "noun_to_verb_ratio": 
            """Ratio of nouns to verbs. Higher ratios, i.e. more nouns, suggest more descriptive, factual rather than action-focused writing. 
            Higher raw values (more nouns to verbs) were associated more strongly with real news than fake news in the training data.
            """,
        
        "cardinal_named_entity_frequency": 
            """Frequency count of numbers and quantities. Higher raw scores signal greater usage of specific details, and were
            associated more with real news in the training data. On the whole, fake news contained fewer numerical facts.""",
        
        "person_named_entity_frequency": 
            """Frequency count of PERSON named entities. Indicates how person-focused the text is. Higher raw scores were more 
            associated with fake news, showing that disinformation campaigns (at least based on the training data),
            are closely tied to attempts to harm a person's reputation through different propaganda campaigns.""",
        
        "nrc_positive_emotion_score": 
            """Measures the positive emotional content using the NRC lexicon. The raw score should be a value between 0 and 1. 
            Higher values (closer to 1) indicate more words associated with positive emotions, 
            and a more positive tone was more closely associated more with real news than fake news in the training data.""",
        
        "nrc_trust_emotion_score": 
            """Based on the number of trust-related words using NRC lexicon. The raw score should be a value between 0 and 1.
            Higher values mean more credibility-focused language, and were more associated with real news than fake news. 
            Fake news contained considerably lower scores for trust-related words in the training data.""",
        
        "flesch_kincaid_readability_score": 
            """U.S. grade level required to understand the text. A higher raw score represents more complex writing, which was 
            associated more with real news in the training data, while, generally, fake news samples relied on simpler language. A score over
            15 means writing is at an advanced, academic level.""",
        
        "difficult_words_readability_score": 
            """Frequency count of complex words not included in the Dall-Chall word list. Higher raw values indicate more sophisticated vocabulary, 
            which was associated more with real news in the training data.""",
        
        "capital_letter_frequency": 
            """Frequency count of capital letters. Higher raw values might indicate more emphasis, or more acronyms and abbreviations. 
            Higher capital letter counts were associated more strongly with real news in training data."""
    }

    warning_message = """
    **Some Guidance on App Usage:** \n 
    - The LIME explanation algorithm used to determine word importance scores functions by removing random words from the text, and then calculating the impact this had on the final prediction.\n  
    - However, the bar charts showing word features pushing towards either the main prediction or against it can carry little meaning **out of context**.\n  
    - Therefore, it is recommended to also inspect how the word appears in the entire **highlighted text**. E.g. the word 'kind' can mean sympathetic and helpful, but it can also be part of the colloquial phrase 'kind of' and thus carry a different meaning!  
    - **Please remember that the final probabilities are a result of interactions between different features, and not the result of one feature alone.** \n 
    """

    warning_message_for_first_two_tabs = """
            **Please remember that the final probabilities are a result of interactions between different features, and not the result of one feature alone.**
            \n Go to the *How it Works* tab for a more detailed explanation of how to interpret the word and semantic feature importance scores.
    """

    @st.cache_resource
    def load_fasttext_model():
        url = "https://drive.google.com/uc?id=1uO8GwNHb4IhqR2RNZqp1K-FdmX6FPnDQ"
        local_path = "/tmp/fasttext_model.bin"
        gdown.download(url, local_path, quiet=False)
        return fasttext.load_model(local_path)

    @st.cache_resource 
    def load_pipeline():
        return joblib.load("all_four_calib_model.pkl")

    @st.cache_resource
    def load_scaler():
        return joblib.load("all_four_standard_scaler.pkl")

    feature_extractor = BasicFeatureExtractor()

    with st.spinner("Loading fake news detection model..."):
        pipeline = load_pipeline()

    with st.spinner("Loading pre-fitted feature scaler..."):
        scaler = load_scaler()

    with st.spinner("Loading fastText embeddings model..."):
        fasttext_model = load_fasttext_model()

    tabs = st.tabs(["Enter News as URL", "Paste in Text Directly", "Key Pattern Visualizations",
                    "Word Clouds: Real vs Fake", "How it Works..."])

    with st.container():
        st.markdown("<style>.block-container {max-width: 2000px;}</style>", unsafe_allow_html=True)

        with tabs[0]:
            st.header("Paste URL to News Text Here")
            url = st.text_area("Enter news URL for classification", placeholder="Paste your URL here...", height=68)
            num_perturbed_samples = st.slider("Select the number of perturbed samples for explanation", 25, 500, 50, 25,
                help="Increasing the number of samples will make the outputted explanations more accurate but may take longer to process!")
            st.write("The more perturbed samples you choose, the more accurate the explanation will be, but it will take longer to output.")
            st.warning(warning_message_for_first_two_tabs, icon='Warning')
            
            if st.button("Classify", key="classify_button_url"): 
                if url.strip():  
                    try:
                        with st.spinner("Extracting news text from URL..."):
                            user_agent = "Mozilla/5.0 (Linux; Android 10; SM-G996U Build/QP1A.190711.020; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Mobile Safari/537.36"
                            config = Config()
                            config.browser_user_agent = user_agent
                            config.request_timeout = 10
                            config.fetch_images = False
                            article = Article(url, config=config)
                            article.download()
                            article.parse()
                            news_text = article.text
                            is_english = detectIfTextIsEnglish(news_text)
                            if is_english == True:
                                analyzeNewsText(news_text, fasttext_model, pipeline, scaler, feature_extractor, num_perturbed_samples, FEATURE_EXPLANATIONS, num_features=50)
                            elif is_english == False:
                                st.error("This text has been detected as non-English. As this model was trained on English news only, please enter an English language text.")
                            else:
                                st.error("Could not extract the news text from the URL, please enter it directly by copying and pasting it in the second tab.")
                    except Exception as e:
                        try:
                            news_text = scrapeWithSoup(url)
                            is_english = detectIfTextIsEnglish(news_text)
                            if is_english == True:
                                analyzeNewsText(news_text, fasttext_model, pipeline, scaler, feature_extractor, num_perturbed_samples, FEATURE_EXPLANATIONS, num_features=50)
                            elif is_english == False:
                                st.error("This text has been detected as non-English. As this model was trained on English news only, please enter an English language text.")
                            else:
                                st.error("Could not extract the news text from the URL, please enter it directly by copying and pasting it in the second tab.")
                        except Exception as e:
                            st.error("Could not extract and analyze the news text. Please try to copy and paste in the text directly in the second tab.")
                else:
                    st.warning("Warning: Please enter some valid news text for classification!")

        with tabs[1]:
            st.header("Paste News Text In Here Directly")
            news_text = st.text_area("Paste the news text for classification", placeholder="Paste your news text here...", height=300)
            num_perturbed_samples = st.slider("Select the number of perturbed samples to use for the explanation", 25, 500, 50, 25,
                help="Warning: Increasing the number of samples will make the outputted explanations more accurate but may take longer to process!")
            st.write("The more perturbed samples you choose, the more accurate the explanation will be, but it will take longer to compute.")
            st.warning(warning_message_for_first_two_tabs, icon='Warning')
            
            if st.button("Classify", key="classify_button_text"):
                if news_text.strip():  
                    try:
                        with st.spinner(f"Analyzing text with {num_perturbed_samples} perturbed samples..."):
                            if detectIfTextIsEnglish(news_text):
                                analyzeNewsText(news_text, fasttext_model, pipeline, scaler, feature_extractor, num_perturbed_samples, FEATURE_EXPLANATIONS, num_features=50)
                            else:
                                st.error("This text has been detected as non-English. As this model was trained on English news only, please enter an English language text.")
                    except Exception as e:
                        st.error(f"Sorry, but there was an error while analyzing the text: {e}")
                else:
                    st.warning("Warning: Please enter some valid news text for classification!")

        with tabs[2]:
            st.header("Key Patterns in the Training Dataset: Real (Blue) vs Fake (Red) News")
            st.write("These visualizations show the main trends and patterns between real and fake news articles in the training data.")
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                st.markdown("<br>", unsafe_allow_html=True) 
                st.subheader("Capital Letter Usage")
                caps_img = plt.imread("all_four_datasets_capitals_bar_chart_real_vs_fake.png")
                st.image(caps_img, caption="Mean number of capital letters in real vs fake news", use_container_width=True)
                st.write("Real news tended to use more capital letters, perhaps due to including more proper nouns and technical acronyms.")
                st.markdown("<br>", unsafe_allow_html=True) 

                st.subheader("Exclamation Point Usage")
                exclaim_img = plt.imread("all_four_datasets_exclamation_points_bar_chart_real_vs_fake.png")
                st.image(exclaim_img, caption="Frequency of exclamation points in real vs fake news", use_container_width=True)
                st.write("Fake news tends to use more exclamation points, possibly suggesting a more sensational and inflammatory writing.")
                st.markdown("<br>", unsafe_allow_html=True) 

                st.subheader("Third Person Pronoun Usage")
                pronouns_img = plt.imread("all_four_datasets_third_person_pronouns_bar_chart_real_vs_fake.png")
                st.image(pronouns_img, caption="Frequency of third-person pronouns in real vs fake news", use_container_width=True)
                st.write("Fake news often uses more third-person pronouns (e.g him, his, her), which could indicate a more 'storytelling' kind of narrative style.")
                st.markdown("<br>", unsafe_allow_html=True) 

                st.subheader("Noun-to-Verb Ratio")
                noun_verb_img = plt.imread("all_four_datasets_noun_to_verb_ratio_bar_chart_real_vs_fake.png")
                st.image(noun_verb_img, caption="Noun-to-Verb Ratio: Real vs Fake News", use_container_width=True)
                st.write("In the training data, real news tended to have slightly more nouns than verbs than fake news.")
                st.markdown("<br>", unsafe_allow_html=True) 

                st.subheader("Emotional Content using NRC Emotion Lexicon")
                emotions_img = plt.imread("all_four_datasets_emotions_bar_chart_real_vs_fake.png")
                st.image(emotions_img, caption="Emotional content comparison between real and fake news", use_container_width=True)
                st.write("Fake news (in this dataset) often showed lower positive emotion scores and fewer trust-based emotion words than real news.")
                st.markdown("<br>", unsafe_allow_html=True) 

                st.subheader("Named Entity PERSON Frequency Counts")
                person_img = plt.imread("all_four_datasets_person_named_entities_bar_chart_real_vs_fake.png")
                st.image(person_img, caption="PERSON named entity count for fake vs real news", use_container_width=True)
                st.write("Fake news (in this dataset) often contained more references to PERSON named entities than real news.")
                st.markdown("<br>", unsafe_allow_html=True) 

                st.subheader("Named Entity CARDINAL (i.e. numbers) Frequency Counts")
                cardinal_img = plt.imread("all_four_datasets_cardinal_named_entities_bar_chart_real_vs_fake.png")
                st.image(cardinal_img, caption="CARDINAL (numbers) named entity count for fake vs real news", use_container_width=True)
                st.write("Fake news tended to contain less numerical data (i.e. lower CARDINAL named entity frequencies) than real news.")
                st.markdown("<br>", unsafe_allow_html=True) 

                st.subheader("Flesch-Kincaid U.S. Readability Grade Level")
                fk_img = plt.imread("all_four_datasets_flesch_kincaid_readability_bar_chart_real_vs_fake.png")
                st.image(fk_img, caption="Flesch-Kincaid avg. U.S. grade level (readability) for fake vs real news", use_container_width=True)
                st.write("Real news tended to have a slightly higher U.S. grade level, indicating more complex language, than fake news.")
                st.markdown("<br>", unsafe_allow_html=True) 

                st.subheader("Difficult Words Score")
                diff_img = plt.imread("all_four_datasets_difficult_words_score_bar_chart_real_vs_fake.png")
                st.image(diff_img, caption="Normalized 'Difficult Words' scores for fake vs real news", use_container_width=True)
                st.write("Real news tended to contain more complex words than fake news.")
                st.markdown("<br>", unsafe_allow_html=True) 

            with st.expander("Details about these visualizations"):
                st.markdown("""
                These charts are created on the basis of a detailed data analysis of four benchmark fake news datasets used for training the model:
                WELFake, Constraint (COVID-19 data), PolitiFact (political news), and GossipCop (entertainment and celebrity news). 
                The charts display the NORMALIZED frequencies, e.g. for exclamation marks and capital use: the 
                raw frequencies have been divided by the text length (in words) to account for the differences in the lengths of the different news texts.
                
                ### Some of the Main Differences between Real News and Fake News Based on the Data Analysis:
                
                - **Capital Letter Frequencies:** Higher frequencies were found in real news, perhaps due to the greater usage of proper nouns and techical acronyms
                - **Third-person Pronoun Frequencies:** Third-person pronouns were more frequenty encountered in fake news in these datasets, suggesting storytelling-like narrative style and person-focused content
                - **Exclamation Point Frequencies:** These were more frequent in fake news too, pointing towards a sensational inflammatory style
                - **Emotion (Trust and Positive) Features:** The words used in fake news tended to have much less positive emotional connotations and reduced trust scores.
                - **Named-Entity (PERSON and CARDINAL) Frequencies:** While fake news contained more PERSON references, real news tended to contain more CARDINAL (number) references
                        to quantitative entities.
                - **Readability Scores:** On the whole, real news tended to contain more complex words and language than fake news
                
                Disclaimer: These patterns were specific to THESE four datasets, but they should be considered in combination with other features
                (i.e. the word feature importance), as well as remembering that more recent fake news may exhibit different trends, particularly
                given the rapid analysis of propaganda and disinformation strategies
                """)

        with tabs[3]:
            st.header("Most Common Named Entities: Exclusive Entities found in Real vs Fake News")
            st.write("These word clouds visualize the most frequent named entities (e.g. people, organizations, countries) in real vs fake news articles in the training data. The size of each word is proportional to how frequently it appears.")
            st.markdown("<br>", unsafe_allow_html=True) 

            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("Named Entities Appearing ONLY in Real News and NOT in Fake News")
                real_cloud_img = plt.imread("combined_four_set_training_data_real_news_named_entities_wordcloud.png")
                st.image(real_cloud_img, caption="Most frequent entities exclusive to real news", use_container_width=True)
            with col2:
                st.subheader("Named Entities Appearing ONLY in Fake News and NOT in Real News")
                fake_cloud_img = plt.imread("combined_four_set_training_data_fake_news_named_entities_wordcloud.png")
                st.image(fake_cloud_img, caption="Most frequent entities exclusive to fake news", use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True) 
            with st.expander("Word Cloud Explanation"):
                st.write("""
                The size of the word reflects how frequently it occurred in each dataset.
                The colors are only used for readability - they don't carry any additional meaning.
                """)

        with tabs[4]:
            st.header("How Does the App Work?")
            st.write("""
            The LIME algorithm (Local Interpretable Model-agnostic Explanations) is used here to explain the
            specific prediction made for a piece of news (i.e. whether the news text is classed as real or fake news).
                    
            Let's get a glimpse into the general intuition behind how this technique works.
            """)
            
            st.subheader("The Main Idea Behind LIME")
            st.write("""
            Whenever this app analyzes a news text, it doesn't just tell you if the news is "fake news" or "real news". The main concept behind
            LIME is to explain which features (e.g. words, certain punctuation patterns) of the text led the model to make the outputted decision.
            As such, highlights WHICH word features, or more high-level semantic and linguistic features (such as use of certain punctuation marks)
            , in the news text led to the outputted classification. Furthermore, the algorithm also outputs the probability of news being fake,
            rather than a simple label, so that you can get an insight into the certainty of the classifier.
            """)
            
            st.subheader("How Does LIME Generate the Explanations?")
            st.write("""
            LIME removes certain words or linguistic features in the news text one-by-one, and runs the trained machine-learning model to see
            how the outputted probabilities change when the text has been slightly changed.

            (a) LIME randomly removes words or linguistic features from the news input
            (b) It then runs the altered versions of the news texts through the classifier and records how much changing these individual features
            has impacted the final prediction
            (c) If changing a specific feature (e.g. emotion score) has a big impact on the final predicted probability, this feature is then assigned a higher importance
            score. The importance scores are then visualized using bar graphs and highlighted texts. Red color-coding means that this feature is associated more
            with fake news, and blue color-coding means this feature makes the text more likely to be real news.
            """)
            
            st.subheader("Which extra features (apart from words) have been included for making predictions?")
            st.write("""
            This model classifies news articles based on the specific features that were found to be the most useful for discriminating 
            between real and fake news based on an extensive exploratory data analysis:

            - Individual words that push a prediction to either real news or fake news
            - Use of punctuation e.g. exclamation marks, capital letters
            - Grammatical patterns such as noun-to-verb ratio
            - Frequencies of PERSON and CARDINAL (number) named entities
            - Trust and positive emotion scores (using the NRC Emo Lexicon)
            - Text readability scores (how hard the text is to read), e.g. how many difficult words are used, U.S. Grade readability level
            """)
            
            with st.expander("Why Were THESE Particular Features Chosen?"):
                st.write("""
                These features were engineered based on a detailed exploratory analysis focusing on the key differences between real and fake news
                over four benchmark datasets: WELFake (general news), Constraint (COVID-19 related health news), PolitiFact (political news),
                and GossipCop (celebrity and entertainment news).
                
                - Fake news is often associated with a more sensational style (e.g. using more exclamation points) than real news, and more "clickbaity" language
                - Real news tends to use more nouns than verbs, as well as more references to numbers, signalling a more factal style
                - Narrative style (e.g. using more third-person pronouns indicates a more "storytelling" style) can also be a key indicator of fake news
                - Text readability and complexity can also help the classifier distinguish between real and fake news,
                as fake news tends to be easier to digest and less challenging.
                """)

            st.warning(warning_message, icon='Warning')
                
            st.subheader("Disclaimer: Limitations of the Model")
            st.markdown("""
                Please bear in mind that the strategies for producing fake news/propaganda are always evolving rapidly, especially due to the rise of generative AI.
                The patterns highlighted here are based on THIS specific training data from four well-known fake news datasets; however,
                they may not apply to newer forms of disinformation!  As a result, it is also strongly recommended to
                use fact-checking and claim-busting websites to check out whether the sources of information are legitimate.
                <br>
                <br>
                The model used to classify fake news here obtained 93% accuracy and F1-score on the training data composed of four different
                dataset from different domains, therefore its predictions are not perfect.
            """, unsafe_allow_html=True)

elif auth_status is False:
    st.error("Incorrect email/password")
elif auth_status is None:
    st.warning("Please log in to continue")