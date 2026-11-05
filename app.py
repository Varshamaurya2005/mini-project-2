import os
import json
import urllib.parse
import threading
import webbrowser
from datetime import datetime
from pathlib import Path
import feedparser
import joblib
import pandas as pd
import requests
import mysql.connector
from flask import Flask, render_template, request, flash, redirect, url_for

from config import MODEL_DIR, RESULTS_DIR, MYSQL_CONFIG

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-truthlens")

MODEL_FILES = {
    "Naive Bayes": "naive_bayes.joblib",
    "Logistic Regression": "logistic_regression.joblib",
    "Linear SVM": "linear_svm.joblib",
    "Random Forest": "random_forest.joblib",
}


def get_connection():
    return mysql.connector.connect(**MYSQL_CONFIG)


def get_best_model_name():
    path = Path(MODEL_DIR) / "best_model.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f).get("best_model", "Naive Bayes")
    return "Naive Bayes"


def fetch_google_rss_news(query, max_results=6):
    encoded_query = urllib.parse.quote(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(rss_url)
    
    articles = []
    for entry in feed.entries[:max_results]:
        source_name = entry.source.get("title", "News Outlet") if hasattr(entry, "source") else "Journalistic Publisher"
        articles.append({
            "title": entry.title,
            "link": entry.link,
            "published": getattr(entry, "published", "Recent"),
            "source": source_name
        })
    return articles


def fetch_google_fact_checks(claim):
    api_key = os.getenv("GOOGLE_FACTCHECK_API_KEY", "")
    if not api_key:
        return []
    
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={urllib.parse.quote(claim)}&key={api_key}"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            claims_data = data.get("claims", [])
            fact_checks = []
            for c in claims_data[:3]:
                for review in c.get("claimReview", []):
                    fact_checks.append({
                        "text": c.get("text", claim),
                        "claimant": c.get("claimant", "Public Claim"),
                        "rating": review.get("textualRating", "Unrated"),
                        "publisher": review.get("publisher", {}).get("name", "Fact Checker"),
                        "url": review.get("url", "#")
                    })
            return fact_checks
    except Exception as e:
        print(f"Fact check API error: {e}")
    return []


def save_history(news_text, input_type, algorithm, ml_prediction, verification_status):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        now = datetime.now()
        query = """
            INSERT INTO search_history 
            (news_text, input_type, algorithm_name, ml_prediction, verification_status, search_date, search_time)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(
            query,
            (
                news_text,
                input_type,
                algorithm,
                ml_prediction,
                verification_status,
                now.date(),
                now.strftime("%H:%M:%S")
            ),
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Database error while saving history: {e}")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect", methods=["GET", "POST"])
def detect():
    model_options = list(MODEL_FILES.keys())

    if request.method == "POST":
        news_text = request.form.get("news_text", "").strip()
        selected_model = request.form.get("algorithm", "Best model")

        if not news_text or len(news_text) < 15:
            flash("Kripya kam se kam 15 characters ka news text enter karein.")
            return redirect(url_for("detect"))

        target_algo = get_best_model_name() if selected_model == "Best model" else selected_model
        model_filename = MODEL_FILES.get(target_algo)

        if not model_filename or not (MODEL_DIR / model_filename).exists():
            flash(f"Model '{target_algo}' nahi mila. Pehle train_models.py run karein.")
            return redirect(url_for("detect"))

        pipeline = joblib.load(MODEL_DIR / model_filename)
        prediction = pipeline.predict([news_text])[0]

        save_history(news_text, "ML Text", target_algo, prediction, "N/A (ML Only)")

        return render_template(
            "result.html",
            algorithm=target_algo,
            prediction=prediction,
            news_text=news_text,
        )

    return render_template("detect.html", models=model_options)


@app.route("/live_news", methods=["GET", "POST"])
def live_news():
    articles = []
    search_query = ""
    if request.method == "POST":
        search_query = request.form.get("query", "").strip()
        if search_query:
            articles = fetch_google_rss_news(search_query, max_results=6)
    return render_template("live_news.html", articles=articles, query=search_query)


@app.route("/verify_news", methods=["GET", "POST"])
def verify_news():
    if request.method == "POST":
        claim = request.form.get("claim", "").strip()
        if not claim or len(claim) < 10:
            flash("Verification ke liye valid claim enter karein.")
            return redirect(url_for("verify_news"))

        best_algo = get_best_model_name()
        pipeline = joblib.load(MODEL_DIR / MODEL_FILES[best_algo])
        ml_prediction = pipeline.predict([claim])[0]

        related_articles = fetch_google_rss_news(claim, max_results=4)
        fact_checks = fetch_google_fact_checks(claim)

        if fact_checks:
            negative_ratings = ["false", "fake", "incorrect", "misleading", "altered"]
            is_debunked = any(any(neg in fc["rating"].lower() for neg in negative_ratings) for fc in fact_checks)
            if is_debunked:
                evidence_status = "Contradicted by Available Evidence"
            else:
                evidence_status = "Supported by Available Evidence"
        elif len(related_articles) >= 2:
            evidence_status = "Supported by Available Evidence (Corroborated by Live Outlets)"
        elif len(related_articles) == 1:
            evidence_status = "Mixed or Conflicting Evidence"
        else:
            evidence_status = "Unverified / Insufficient Information"

        save_history(claim, "Live Claim", best_algo, ml_prediction, evidence_status)

        return render_template(
            "verify_result.html",
            claim=claim,
            ml_algo=best_algo,
            ml_pred=ml_prediction,
            evidence_status=evidence_status,
            articles=related_articles,
            fact_checks=fact_checks
        )

    return render_template("verify_news.html")


@app.route("/comparison")
def comparison():
    metrics_path = RESULTS_DIR / "metrics.csv"
    metrics = []
    best_algo = get_best_model_name()

    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        metrics = df.to_dict(orient="records")

    return render_template("comparison.html", metrics=metrics, best=best_algo)


@app.route("/history")
def history():
    rows = []
    error = None
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT id, news_text, algorithm_name, ml_prediction, verification_status, search_date, search_time FROM search_history ORDER BY id DESC LIMIT 50"
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
    except Exception as e:
        error = f"Database se records fetch nahi ho paye: {e}"

    return render_template("history.html", rows=rows, error=error)


def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")


if __name__ == "__main__":
    threading.Timer(1.2, open_browser).start()
    app.run(debug=False, host="127.0.0.1", port=5000)