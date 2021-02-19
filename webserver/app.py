from flask_bootstrap import Bootstrap
from flask import Flask, render_template, Response, request, url_for, redirect, flash
from werkzeug.utils import secure_filename
import os
import subprocess
import time
import sys
from webserver_utils import (
    UPLOAD_FOLDER,
    ensure_folder,
    get_current_time,
    random_string,
    get_predictions,
    generate_report_dict_list,
)
import pandas as pd
import jinja2


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
        out = get_predictions(
            df,
            clf_dir="./best_model_20210211-031248.pkl",
            selected_features="./featureid_best_model_20210211-031248.npy",
        )
        report_d_l = generate_report_dict_list(out)

        print(report_d_l)
    return render_template("result.html", items=report_d_l, filename=filename)


@app.route("/positive", methods=["GET", "POST"])
def positive():
    with open("./data/sample_pos.csv") as f:
        csv = f.read()
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=positive.csv"},
    )


@app.route("/negative", methods=["GET", "POST"])
def negative():
    with open("./data/sample_neg.csv") as f:
        csv = f.read()
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=negative.csv"},
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)