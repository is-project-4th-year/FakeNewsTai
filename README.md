
---

# TAIEYE Fake News Detection System

![TAIEYE Logo](assets/images/taieye_logo.jpg)

## About TAIEYE

**TAIEYE** is a real-time, explainable fake news detection system designed to identify misinformation across multiple domains, including politics, health, and entertainment.
The system integrates classical machine learning, linguistic feature engineering, and explainable AI (XAI) to deliver transparent and trustworthy predictions.

This repository provides tools to:

* Detect fake news using a calibrated stacked ensemble model.
* Generate interpretable explanations via LIME.
* Offer an intuitive Streamlit-based user interface.
* Support secure authentication via Firebase.

---

## Features

* **Streamlit Interface**
  Responsive and user-friendly dashboard for text or URL-based analysis.

* **Hybrid Input Support**
  Accepts raw text or automatically extracts content from URLs.

* **Explainable AI**
  Uses LIME to highlight influential words and linguistic features.

* **Machine Learning Ensemble**
  Logistic Regression, XGBoost, LightGBM, and Random Forest stacked and calibrated.

* **Secure User Authentication**
  Integrated Firebase Authentication for controlled access.

---

## Technologies Used

| Technology                      | Purpose                          |
| ------------------------------- | -------------------------------- |
| **Python**                      | Core development                 |
| **Streamlit**                   | Frontend and UI rendering        |
| **Scikit-learn**                | Classical ML & ensemble stacking |
| **XGBoost / LightGBM**          | Gradient boosting models         |
| **fastText**                    | Word embeddings                  |
| **SpaCy**                       | NLP preprocessing                |
| **LIME**                        | Explainability                   |
| **Firebase Admin SDK**          | Secure authentication            |
| **BeautifulSoup / newspaper3k** | URL text extraction              |

---

## How to Run the Project

```bash
# 1. Clone the repository
git clone https://github.com/is-project-4th-year/FakeNewsTai.git
cd FakeNewsTai

# 2. Create and activate a Python 3.9 virtual environment
python3.9 -m venv venv
source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Start the Streamlit application
streamlit run app.py
# The app will open at http://localhost:8501/
```

---

## Firebase Setup

1. Create a Firebase project at the Firebase Console.
2. Enable **Email/Password Authentication**.
3. Download your admin SDK JSON file.
4. Place it in the project root (ensure it is added to `.gitignore`).

---

## Achievements & System Performance

* Achieves **93% accuracy**, **92% precision**, **94% recall**, and **0.98 ROC-AUC** across a multi-domain test set.
* Generates **real-time LIME explanations** for every prediction.
* Fully compatible with **Apple Silicon (M1/M2/M3)** using CPU-only execution.
* Supports rapid inference with an average latency of **<1.5 seconds**.

---

## Contribution Guidelines

Contributions to improve the system, models, or interface are welcome.

```bash
# 1. Fork the repository

# 2. Create a new branch
git checkout -b feature-branch

# 3. Implement your changes

# 4. Commit using clear, descriptive messages
git commit -m "Added new feature: <description>"

# 5. Push to GitHub and open a pull request
```

---

## License

This project is licensed under the MIT License.
Refer to the **LICENSE** file for details.

---

## Contact

Email: **[tevin.omondi@strathmore.edu](mailto:tevin.omondi@strathmore.edu)**
GitHub: **[https://github.com/Tevin-O](https://github.com/Tevin-O)**

---


