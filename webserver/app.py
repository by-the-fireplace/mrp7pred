from flask_bootstrap import Bootstrap
from flask import Flask, render_template, Response, request, url_for, redirect
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

bootstrap = Bootstrap(app)

## uploading specs ##
UPLOAD_FOLDER = "/"
ALLOWED_EXTENSIONS = set(["csv"])


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1] in ALLOWED_EXTENSIONS


# @app.route("/home", methods=["GET", "POST"])
@app.route("/", methods=["GET", "POST"])
def home():
    ## file uploading stuff
    if request.method == "POST":
        file = request.files["file"]
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            return redirect(url_for("/", filename=filename))
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