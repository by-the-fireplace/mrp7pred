from flask_bootstrap import Bootstrap
from flask import Flask, render_template, Response, request, url_for, redirect, flash
from werkzeug.utils import secure_filename
import os
import subprocess
import time
import sys
from utils import UPLOAD_FOLDER, ensure_folder, get_current_time, random_string
import pandas as pd


app = Flask(__name__)

bootstrap = Bootstrap(app)

current_data = ""

# @app.route("/home", methods=["GET", "POST"])
@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("base.html")


@app.route("/run", methods=["GET", "POST"])
def run():
    ensure_folder(UPLOAD_FOLDER)
    ts = get_current_time()
    rs = random_string(10)
    random_folder = f"{ts}_{rs}"
    ensure_folder(f"{UPLOAD_FOLDER}/{random_folder}")
    app.config["UPLOAD_FOLDER"] = f"{UPLOAD_FOLDER}/{random_folder}"

    if request.method == "POST":
        file = request.files["csv_file"]
        filename = secure_filename(file.filename)
        task_name = f"{ts}_{rs}_{filename}"
        # current_data = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        # print(current_data)
        df = pd.read_csv(file)

    return render_template("run.html", data=rs)


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


if __name__ == "__main__":
    app.run(debug=True)