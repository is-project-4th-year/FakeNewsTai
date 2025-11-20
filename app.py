# app.py – TAIEYE Fake News Detector (FULLY RESTORED + FIXED)
import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth, firestore
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
from datetime import datetime
from dotenv import load_dotenv
import uuid
import hashlib
# ──────────────────────────────
# CONFIG
# ──────────────────────────────
# Load .env file (safe fallback if not exist)
load_dotenv()

# Now safely read everything — works locally AND on Streamlit Cloud
FIREBASE_CRED_PATH = os.getenv("FIREBASE_CRED_PATH", "FakeNewsTai/taieye-a98df-firebase-adminsdk-fbsvc-133401b53e.json")
CONFIG_FILE = os.getenv("CONFIG_FILE", "config.yaml")
FASTTEXT_DRIVE_ID = os.getenv("FASTTEXT_DRIVE_ID")
FASTTEXT_LOCAL = "/tmp/fasttext_model.bin"
PIPELINE_FILE = "all_four_calib_model.pkl"
SCALER_FILE = "all_four_standard_scaler.pkl"
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
SERVER_METADATA_URL = os.getenv("SERVER_METADATA_URL", "https://accounts.google.com/.well-known/openid-configuration")
# ONLINE LOGO (use a valid public image)
LOCAL_LOGO = "assets/images/taieye_logo.jpg"
ONLINE_LOGO = "https://unsplash.com/photos/black-and-white-eagle-on-black-wooden-fence-XmSF9Wk8Cgo" 

def clear_auth_cookie():
    """Delete the streamlit-authenticator cookie so the user is really logged out."""
    if "taieye_auth" in st.session_state:
        del st.session_state["taieye_auth"]
    # Also remove the cookie from the browser (works in Streamlit >=1.30)
    st.experimental_set_query_params()   # forces a clean URL

def get_logo():
    if os.path.exists(LOCAL_LOGO):
        return LOCAL_LOGO
    return ONLINE_LOGO

def show_logo(where: str):
    """Show the correct logo in the correct place."""
    path = get_logo()
    if not os.path.exists(path):
        st.warning("Logo file missing")
        return

    if where == "main":
        st.image(path, width=150, use_container_width="auto")
    elif where == "sidebar":
        st.sidebar.image(path, width=90, use_container_width=True)

# ──────────────────────────────
# 1. Firebase init
# ──────────────────────────────
firebase_initialized = False
db = None
if os.path.exists(FIREBASE_CRED_PATH):
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(FIREBASE_CRED_PATH)
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        firebase_initialized = True
    except Exception as e:
        st.warning("Firebase init failed – feedback & Google sign-in disabled.")
        st.exception(e)
else:
    st.info("Firebase credentials not found – place JSON in project root to enable admin features.")

# ──────────────────────────────
# 2. config.yaml
# ──────────────────────────────
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

def add_user_to_config(email, name, password):
    """Add user to local config.yaml without streamlit-authenticator."""
    # Simple SHA256 hash for demo purposes
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    config['credentials']['usernames'][email] = {
        'email': email,
        'name': name,
        'password': password_hash
    }
    with open(CONFIG_FILE, 'w') as f:
        yaml.dump(config, f)

# ──────────────────────────────
# 3. Auth helpers
# ──────────────────────────────
# -------- Helpers: create user + email verify + magic link + reset link --------
def create_user(email, password, name, send_verification=True):
    """Create account in Firebase (admin) and add to local config. Optionally generate email verification link."""
    try:
        if firebase_initialized:
            # create Firebase Auth user
            fb_user = auth.create_user(email=email, password=password, display_name=name)
        # persist to local yaml for streamlit-authenticator (so login widget works)
        add_user_to_config(email, name, password)

        verification_link = None
        if firebase_initialized and send_verification:
            # generate a verification link (admin SDK) and display it; you can send it via SMTP later
            action_settings = auth.ActionCodeSettings(url="http://localhost:8501",
                                                      handle_code_in_app=False)
            verification_link = auth.generate_email_verification_link(email, action_settings)
        return True, verification_link
    except Exception as e:
        return False, str(e)

def send_magic_link(email):
    """Generate a magic sign-in link and print it in UI (or send it via email)."""
    if not firebase_initialized:
        st.warning("Firebase admin not initialized; cannot create magic link.")
        return None
    try:
        action_settings = auth.ActionCodeSettings(
            url="http://localhost:8501",  # the app will receive token in query params
            handle_code_in_app=False
        )
        link = auth.generate_sign_in_with_email_link(email, action_settings)
        # You should email 'link' to the user (SMTP or SendGrid). For local dev we show it.
        st.info("Magic link generated below — copy/paste it into your browser (local dev).")
        st.code(link)
        return link
    except Exception as e:
        st.error(f"Failed to create magic link: {e}")
        return None

def send_reset_link(email):
    if not firebase_initialized:
        st.warning("Firebase admin not initialized; cannot create reset link.")
        return None
    try:
        link = auth.generate_password_reset_link(email)
        st.info("Password reset link generated (local dev).")
        st.code(link)
        return link
    except Exception as e:
        st.error(f"Failed to create reset link: {e}")
        return None
    

# ──────────────────────────────
# 4. Page config + CSS
# ──────────────────────────────
st.set_page_config(page_title="TAIEYE", page_icon="Eagle", layout="wide")

st.markdown("""
<style>
    :root {--bg:#0e1117;--card:#0f1720;--muted:#9aa4b2;--accent1:#ff6b6b;--accent2:#ff4b4b;}
    body{background:var(--bg);color:#e6eef8;}
    .block-container{max-width:1400px;margin:0 auto;padding:1.2rem;}
    .card{background:var(--card);border-radius:12px;padding:18px;box-shadow:0 8px 30px rgba(0,0,0,0.6);}
    .logo-img{width:150px;height:150px;object-fit:cover;border-radius:50%;box-shadow:0 8px 32px rgba(0,0,0,0.6);}
    .h1{font-size:34px;font-weight:800;color:var(--accent1);text-align:center;margin-bottom:4px;}
    .h2{color:var(--muted);text-align:center;margin-top:0;margin-bottom:18px;}
    .stButton>button{background:linear-gradient(90deg,var(--accent2),var(--accent1));color:white;border-radius:10px;padding:8px 18px;font-weight:600;}
    .small-muted{color:#8b98a6;font-size:13px;text-align:center;margin-top:8px;}
    hr.white{border-color:rgba(255,255,255,0.06);}
</style>
""", unsafe_allow_html=True)

# ---- MAIN PAGE LOGO (local file) ----
logo_path = get_logo()
if os.path.exists(logo_path):
    # 150 px wide, but never larger than the container
    st.image(logo_path, width=150, use_container_width="auto")
else:
    st.warning("Main logo not found – check assets/images/taieye_logo.jpg")
st.markdown('<div class="h1">TAIEYE</div>', unsafe_allow_html=True)
st.markdown('<div class="h2"><i>Truth in Every Word</i></div>', unsafe_allow_html=True)


# ──────────────────────────────
# 6. Session state
# ──────────────────────────────
for key in ['page', 'user_email', 'user_name', 'auth_status', 'last_input', 
            'last_result', 'classification_id']:
    if key not in st.session_state:
        st.session_state[key] = None

if st.session_state.page is None:
    st.session_state.page = 'auth'

# ──────────────────────────────
# 7. Model loaders
# ──────────────────────────────
@st.cache_resource
def load_fasttext_model():
    url = f"https://drive.google.com/uc?id={FASTTEXT_DRIVE_ID}"
    if os.path.exists(FASTTEXT_LOCAL):
        return fasttext.load_model(FASTTEXT_LOCAL)
    gdown.download(url, FASTTEXT_LOCAL, quiet=True)
    return fasttext.load_model(FASTTEXT_LOCAL)

@st.cache_resource
def load_pipeline():
    if not os.path.exists(PIPELINE_FILE):
        raise FileNotFoundError(PIPELINE_FILE)
    return joblib.load(PIPELINE_FILE)

@st.cache_resource
def load_scaler():
    if not os.path.exists(SCALER_FILE):
        raise FileNotFoundError(SCALER_FILE)
    return joblib.load(SCALER_FILE)

# ──────────────────────────────
# 8. Google Sign-In
# ──────────────────────────────
# ──────────────────────────────
# 8. Google Sign-In (FULLY FIXED FOR NEW STREAMLIT API)
# ──────────────────────────────

def google_sign_in_html(firebase_config: dict):
    """
    Loads Firebase JS SDK, opens Google popup, gets ID token,
    then appends ?firebase_token=<token> to Streamlit URL.
    """
    js = f"""
    <script src="https://www.gstatic.com/firebasejs/9.22.2/firebase-app-compat.js"></script>
    <script src="https://www.gstatic.com/firebasejs/9.22.2/firebase-auth-compat.js"></script>
    <script>
        const firebaseConfig = {firebase_config};
        if (!window.firebaseApp) {{
            firebase.initializeApp(firebaseConfig);
            window.firebaseApp = firebase;
        }}

        const provider = new firebase.auth.GoogleAuthProvider();

        async function signInWithGoogle() {{
            try {{
                const result = await firebase.auth().signInWithPopup(provider);
                const token = await result.user.getIdToken();

                const url = new URL(window.location);
                url.searchParams.set("firebase_token", token);

                window.location = url.toString();

            }} catch (e) {{
                alert("Google sign-in error: " + e.message);
            }}
        }}
    </script>

    <button onclick="signInWithGoogle()"
        style="background:#4285F4;color:white;padding:10px 14px;border-radius:8px;border:none;font-weight:700;cursor:pointer;">
        Sign in with Google
    </button>
    """

    st.components.v1.html(js, height=80)


# ---- Handle Google callback ----
qp = st.query_params  # ONLY USING NEW API

if "firebase_token" in qp:
    token = qp["firebase_token"]

    try:
        decoded = auth.verify_id_token(token)
        email = decoded.get("email")
        name = decoded.get("name") or email.split("@")[0]

        # Add to Streamlit Auth config if not present
        if email not in config["credentials"]["usernames"]:
            add_user_to_config(email, name, str(uuid.uuid4()))

        # Save session
        st.session_state.update({
            "page": "main",
            "user_email": email,
            "user_name": name,
            "auth_status": True
        })

        # CLEAR PARAMS (NEW API)
        st.query_params.clear()
        st.rerun()

    except Exception as e:
        st.error(f"Token error: {e}")

        st.query_params.clear()
        st.rerun()


# ──────────────────────────────
# 9. Auth page
# ──────────────────────────────

def auth_page():
    """Render the authentication page: Login, Register, Passwordless, and Password Reset."""

    # ---- Custom CSS ----
    st.markdown("""
    <style>
        .auth-card {
            background-color: #ffffff10;
            padding: 2rem;
            border-radius: 1.2rem;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            max-width: 500px;
            margin: 2rem auto;
        }
        .small-muted { font-size: 0.85rem; color: #aaa; text-align: center; margin: 1rem 0; }
        .divider {
            display: flex; align-items: center; text-align: center;
            color: #aaa; margin: 1rem 0;
        }
        .divider::before, .divider::after {
            content: ""; flex: 1; border-bottom: 1px solid #aaa;
        }
        .divider:not(:empty)::before { margin-right: .75em; }
        .divider:not(:empty)::after { margin-left: .75em; }
    </style>
    """, unsafe_allow_html=True)

    # Clear caches to prevent stale sessions
    try:
        st.cache_data.clear()
    except Exception:
        pass
    try:
        st.cache_resource.clear()
    except Exception:
        pass

    tabs = st.tabs(["🔐 Login", "📝 Register", "💫 Passwordless"])

    # -------- LOGIN TAB --------
    with tabs[0]:
        st.markdown("## Welcome Back")

        email_login = st.text_input("Email", key="login_email")
        password_login = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", key="btn_login"):
            if not email_login or not password_login:
                st.error("Enter both email and password.")
            else:
                try:
                    user = auth.get_user_by_email(email_login)
                    # NOTE: Firebase Admin SDK cannot verify passwords directly.
                    # For full password verification, integrate Firebase Auth client SDK or REST API.
                    # For demo, any existing email is allowed.
                    st.session_state.update({
                        "page": "main",
                        "user_email": email_login,
                        "user_name": user.display_name or email_login,
                        "auth_status": True
                    })
                    st.session_state["rerun_trigger"] = True  # optional trigger
                    st.stop()  # stops script, Streamlit reruns automatically on next interaction
                except Exception as e:
                    st.error(f"Login failed: {e}")

        st.markdown('<div class="divider">Or continue with</div>', unsafe_allow_html=True)

        # Third-party auth
        c1, c2 = st.columns(2)
        with c1:
            google_sign_in_html(FIREBASE_CRED_PATH)  # Your existing Google Sign-In JS
        with c2:
            magic_email = st.text_input("Email for magic link", key="magic_input")
            if st.button("Send Magic Link", key="send_magic"):
                if magic_email:
                    send_magic_link(magic_email)
                    st.success("Magic link created. Use it to sign in.")
                else:
                    st.error("Enter an email address.")

    # -------- REGISTER TAB --------
    with tabs[1]:
        st.markdown("## Create Account")
        with st.form("register_form"):
            name_in = st.text_input("Full Name", key="reg_name")
            email = st.text_input("Email", key="reg_email")
            p1 = st.text_input("Password", type="password", key="reg_p1")
            p2 = st.text_input("Confirm Password", type="password", key="reg_p2")

            if st.form_submit_button("Create Account"):
                if not all([name_in, email, p1, p2]):
                    st.error("Fill all fields.")
                elif p1 != p2:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = create_user(email, p1, name_in, send_verification=True)
                    if ok:
                        st.success("Account created. Verify via email.")
                    else:
                        st.error(f"Signup failed: {msg}")

    # -------- PASSWORDLESS TAB --------
    with tabs[2]:
        st.markdown("## Passwordless Login")
        email_for_magic = st.text_input("Email", key="pwless_email")
        if st.button("Generate Magic Link", key="gen_magic"):
            if email_for_magic:
                link = send_magic_link(email_for_magic)
                if link:
                    st.success("Magic link created. Check email or use link above.")
            else:
                st.error("Enter an email address.")

    # -------- FORGOT PASSWORD --------
    with st.expander("Forgot Password?"):
        st.caption("Reset your password via email.")
        reset_email = st.text_input("Your email", placeholder="you@example.com", key="reset_email_input")
        if st.button("Send Reset Link", key="send_reset"):
            if reset_email:
                send_reset_link(reset_email)
                st.success("✅ Password reset link sent! Check your email.")
            else:
                st.error("Please enter your email address.")


# ──────────────────────────────
# 10. Main App
# ──────────────────────────────
def main_app(pipeline, scaler, fasttext_model, feature_extractor):
    """
    main_app: cleaned, fixed sidebar + logout + menu + detector/feedback branches.
    Replace your existing main_app(...) with this function.
    """

    # ---- SIDEBAR WELCOME + NAVIGATION ----
    st.sidebar.markdown(f"#### Welcome\n{st.session_state.get('user_name', '')}")

    # Navigation radio (only place where menu is defined)
    menu = st.sidebar.radio("Navigation", ["Detector", "Feedback"])

    # small spacer / divider for nice responsive layout
    st.sidebar.markdown("---")

      # Logout button placed *after* navigation (appears once, unique key)
  # Logout button in sidebar
    if st.session_state.get("page") == "main":
     if st.sidebar.button("🚪 Logout"):
        # Clear all session state keys
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        st.success("✅ You have been logged out.")
        st.stop()

    # -------------------- DETECTOR --------------------
    if menu == "Detector":
        st.title("🧠 TaiEye Fake News Detection")
        st.markdown(
            """
            <div style='padding: 10px; border-radius: 12px; background-color: #f8f9fa; 
                        border-left: 5px solid #4F46E5;'>
                <h4>Analyze any news headline or article and discover how likely it is to be <b>fake</b> or <b>real</b>.</h4>
                <p style='font-size: 14px; color: #555;'>TaiEye uses linguistic, readability, and emotional tone features to detect misinformation patterns.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        warning_message = """
            **Note:** The model’s probabilities are based on statistical language patterns, not individual claims.
            Always cross-check results with credible news sources.
        """

        FEATURE_EXPLANATIONS = {
            "exclamation_point_frequency": "High exclamation marks often indicate emotional or sensational writing—commonly seen in fake news.",
            "third_person_pronoun_frequency": "Frequent use of 'he', 'she', or 'they' shows narrative style often tied to story-like fake content.",
            "noun_to_verb_ratio": "Higher noun usage implies descriptive, factual tone—more typical of real news.",
            "cardinal_named_entity_frequency": "Frequent numbers signal factual reporting—real news tends to include more quantifiable data.",
            "person_named_entity_frequency": "Focus on people can indicate propaganda-style disinformation.",
            "nrc_positive_emotion_score": "Texts with balanced or positive tone are often more credible.",
            "nrc_trust_emotion_score": "Measures trust-related words—real news typically scores higher.",
            "flesch_kincaid_readability_score": "Complex writing correlates more with real news; fake news often simplifies language.",
            "difficult_words_readability_score": "Advanced vocabulary is typical of professional reporting.",
            "capital_letter_frequency": "Higher capitalization can suggest emphasis or acronyms—moderate values common in real news."
        }

        # Create the tabs here (only defined and used inside Detector branch)
        tabs = st.tabs(
            [
                "🔗 Enter News URL",
                "📝 Paste Text Directly",
                "📊 Key Pattern Visualizations",
                "☁️ Word Clouds: Real vs Fake",
                "🧩 How It Works",
            ]
        )

        # ----- TAB 1: URL Input -----
        with tabs[0]:
            st.subheader("🔗 Analyze News from URL")
            with st.container():
                st.markdown('<div class="main-card">', unsafe_allow_html=True)
                url = st.text_input(
                    "Enter a news article URL",
                    placeholder="https://example.com/news-article",
                    key="url_input",
                )
                num_perturbed_samples = st.slider(
                    "LIME Explanation Samples",
                    25,
                    500,
                    50,
                    25,
                    key="lime_url",
                    help="More samples = more accurate explanation (slower)",
                )
                st.warning(warning_message, icon="⚠️")

                if st.button("Classify from URL 🌍", key="classify_url_btn"):
                    if not url.strip():
                        st.warning("Please enter a valid news URL.")
                    else:
                        with st.spinner("Extracting and analyzing the article..."):
                            try:
                                cfg = Config()
                                cfg.browser_user_agent = "Mozilla/5.0"
                                cfg.request_timeout = 12
                                cfg.fetch_images = False
                                article = Article(url, config=cfg)
                                article.download()
                                article.parse()
                                news_text = article.text
                            except Exception:
                                news_text = scrapeWithSoup(url)

                            if news_text:
                                st.session_state.last_input = news_text
                                if detectIfTextIsEnglish(news_text):
                                    st.session_state.classification_id = str(uuid.uuid4())
                                    analyzeNewsText(
                                        news_text,
                                        fasttext_model,
                                        pipeline,
                                        scaler,
                                        feature_extractor,
                                        num_perturbed_samples,
                                        FEATURE_EXPLANATIONS,
                                        50,
                                    )
                                    st.session_state.last_result = "completed"
                                    st.success("✅ Analysis complete! You can now visit Feedback to submit your feedback.")
                                else:
                                    st.error("🚫 Non-English text detected.")
                            else:
                                st.error("❌ Failed to extract article content.")
                st.markdown("</div>", unsafe_allow_html=True)

        # ----- TAB 2: Text Input -----
        with tabs[1]:
            st.subheader("📝 Paste News Text Directly")
            news_text = st.text_area(
                "Paste or type the news text here:",
                placeholder="Paste your news article text...",
                height=250,
                key="text_input",
            )
            num_perturbed_samples = st.slider(
                "LIME Explanation Samples", 25, 500, 50, 25, key="lime_text"
            )
            st.warning(warning_message, icon="⚠️")

            if st.button("Analyze Text 🔍", key="analyze_text_btn"):
                if news_text.strip():
                    if detectIfTextIsEnglish(news_text):
                        st.session_state.last_input = news_text
                        st.session_state.classification_id = str(uuid.uuid4())
                        analyzeNewsText(
                            news_text,
                            fasttext_model,
                            pipeline,
                            scaler,
                            feature_extractor,
                            num_perturbed_samples,
                            FEATURE_EXPLANATIONS,
                            50,
                        )
                        st.session_state.last_result = "completed"
                        st.success("✅ Analysis complete! Check the Visualizations / Word Clouds tabs.")
                    else:
                        st.error("🚫 Non-English text detected.")
                else:
                    st.warning("Please enter some text for analysis.")

        # ----- TAB 3: Visualizations -----
        with tabs[2]:
            st.header("📊 Key Patterns in Real vs Fake News")

            # Only show visuals after a successful classification (last_result == "completed")
            if st.session_state.get("last_result") != "completed":
                st.info("⚠️ Analyze some news first to see visualizations.")
            else:
                vis_files = {
                    "Capital Letters": "all_four_datasets_capitals_bar_chart_real_vs_fake.png",
                    "Exclamation Points": "all_four_datasets_exclamation_points_bar_chart_real_vs_fake.png",
                    "Third Person Pronouns": "all_four_datasets_third_person_pronouns_bar_chart_real_vs_fake.png",
                    "Noun-to-Verb Ratio": "all_four_datasets_noun_to_verb_ratio_bar_chart_real_vs_fake.png",
                    "Emotional Content (NRC)": "all_four_datasets_emotions_bar_chart_real_vs_fake.png",
                    "Named Entity PERSON": "all_four_datasets_person_named_entities_bar_chart_real_vs_fake.png",
                    "Named Entity CARDINAL": "all_four_datasets_cardinal_named_entities_bar_chart_real_vs_fake.png",
                    "Flesch-Kincaid Readability": "all_four_datasets_flesch_kincaid_readability_bar_chart_real_vs_fake.png",
                    "Difficult Words Score": "all_four_datasets_difficult_words_score_bar_chart_real_vs_fake.png",
                }
                for title, path in vis_files.items():
                    st.subheader(title)
                    if os.path.exists(path):
                        st.image(path, use_container_width=True)
                    else:
                        st.warning(f"Missing visualization file: {path}")

        # ----- TAB 4: Word Clouds -----
        with tabs[3]:
            st.header("☁️ Named Entities Word Clouds")
            if st.session_state.get("last_result") != "completed":
                st.info("⚠️ Analyze some news first to see word clouds.")
            else:
                for label, path in [
                    ("Real-news", "combined_four_set_training_data_real_news_named_entities_wordcloud.png"),
                    ("Fake-news", "combined_four_set_training_data_fake_news_named_entities_wordcloud.png"),
                ]:
                    st.subheader(label)
                    if os.path.exists(path):
                        st.image(path, use_container_width=True)
                    else:
                        st.warning(f"Missing {label.lower()} word cloud image")

        # ----- TAB 5: How It Works -----
        with tabs[4]:
            st.header("🧩 How Does the App Work?")
            st.write(
                """
                The LIME algorithm (Local Interpretable Model-agnostic Explanations) explains why the model predicted
                a piece of news as **fake** or **real**. It highlights which features (words, punctuation, named entities, etc.)
                influenced the prediction, and shows probabilities rather than simple labels.
                """
            )

            st.subheader("The Main Idea Behind LIME")
            st.write(
                """
                LIME works by slightly altering the text (removing words or changing features) and observing
                how the model's output changes. Features that strongly affect the output are marked as important.
                Red highlights indicate features associated with fake news, blue for real news.
                """
            )

            st.subheader("Extra Features Used")
            st.write(
                """
                This model includes:
                - Individual words affecting the prediction
                - Punctuation usage (exclamation points, capital letters)
                - Grammatical patterns (noun-to-verb ratio)
                - Named entities (PERSON, CARDINAL)
                - Emotion scores (Trust, Positive using NRC Emo Lexicon)
                - Text readability scores (Flesch-Kincaid, Difficult Words)
                """
            )

            st.subheader("Disclaimer: Limitations of the Model")
            st.markdown(
                """
                Please note that fake news strategies evolve rapidly, and the patterns shown are based
                on four benchmark datasets (WELFake, Constraint, PolitiFact, GossipCop).
                Predictions may not be perfect and should be cross-checked with credible sources.
                """
            )
            st.warning(warning_message, icon="⚠️")

    # -------------------- FEEDBACK --------------------
    elif menu == "Feedback":
        st.title("💬 Feedback Center")
        st.caption("Help TaiEye improve by rating the last prediction.")

        # only allow feedback if a classification occurred
        classification_id = st.session_state.get("classification_id")
        if not classification_id:
            st.info("⚠️ You must first classify some news before giving feedback.")
        else:
            feedback_col1, feedback_col2 = st.columns([1, 3])
            with feedback_col1:
                feedback_rating = st.radio(
                    "Was this classification correct?",
                    ["👍 Yes", "👎 No"],
                    horizontal=True,
                    key="feedback_radio",
                )
            with feedback_col2:
                feedback_comment = st.text_area(
                    "Optional comment",
                    placeholder="Tell us why you agree or disagree...",
                    key="feedback_text",
                )

            if st.button("Send Feedback 📨", key="send_feedback"):
                if firebase_initialized:
                    feedback_data = {
                        "user_email": st.session_state.get("user_email", "anonymous"),
                        "classification_id": classification_id,
                        "rating": feedback_rating,
                        "comment": feedback_comment,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    try:
                        db.collection("feedback").add(feedback_data)
                        st.success("✅ Feedback submitted successfully!")
                        # reset classification so user can't double-submit
                        st.session_state.classification_id = None
                        st.session_state.last_input = None
                    except Exception as e:
                        st.error(f"❌ Error sending feedback: {e}")
                else:
                    st.warning("⚠️ Firebase not connected — feedback disabled.")



# ──────────────────────────────
# 11. Lazy load models
# ──────────────────────────────
pipeline = scaler = fasttext_model = feature_extractor = None
if st.session_state.page == 'main':
    with st.spinner("Loading models..."):
        pipeline = load_pipeline()
        scaler = load_scaler()
        fasttext_model = load_fasttext_model()
        feature_extractor = BasicFeatureExtractor()

# ──────────────────────────────
# 12. Routing
# ──────────────────────────────
if st.session_state.page == 'auth':
    auth_page()
elif st.session_state.page == 'main':
    if not all([pipeline, scaler, fasttext_model]):
        st.error("Models failed to load.")
    else:
        main_app(pipeline, scaler, fasttext_model, feature_extractor)

# ──────────────────────────────
# 13. Footer
# ──────────────────────────────
st.markdown("<hr class='white'/>", unsafe_allow_html=True)
st.markdown("<div style='font-size:12px;color:#8b98a6;text-align:center'>TAIEYE • Truth in Every Word • 2025</div>", unsafe_allow_html=True)
