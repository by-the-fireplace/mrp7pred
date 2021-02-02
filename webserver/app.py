from flask_bootstrap import Bootstrap
from flask import Flask

app = Flask(__name__)

bootstrap = Bootstrap(app)

@app.route("/home", methods=["GET", "POST"])