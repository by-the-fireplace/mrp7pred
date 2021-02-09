from flask_bootstrap import Bootstrap
from flask import Flask, render_template, Response

app = Flask(__name__)

bootstrap = Bootstrap(app)


# @app.route("/home", methods=["GET", "POST"])
@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("base.html")


@app.route("/positive", methods=["GET", "POST"])
def positive():
    with open("./data/positive.csv") as f:
        csv = f.read()
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=positive.csv"},
    )


@app.route("/negative", methods=["GET", "POST"])
def negative():
    with open("./data/negative.csv") as f:
        csv = f.read()
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=negative.csv"},
    )