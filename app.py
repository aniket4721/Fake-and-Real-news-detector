from flask import Flask, render_template, request
import joblib
import re
import string

app = Flask(__name__)

vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

def clean_text(text):
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    return text.lower()

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        news = request.form["news"]
        cleaned = clean_text(news)
        transformed = vectorizer.transform([cleaned])
        prediction = model.predict(transformed)

        if prediction[0] == 1:
            result = "This News is REAL ✅"
        else:
            result = "This News is FAKE ❌"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
