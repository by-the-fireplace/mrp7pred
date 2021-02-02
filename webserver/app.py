from flask_bootstrap import Bootstrap
from flask import Flask, render_template

app = Flask(__name__)

bootstrap = Bootstrap(app)


# @app.route("/home", methods=["GET", "POST"])
@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("home.html")
