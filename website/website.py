from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
    # A super-simple page render; all copy lives in the template.
    return render_template("index.html")

# Render.com will look for "gunicorn website:app" by default if you set it that way
# You can still run locally with:  python website.py
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
