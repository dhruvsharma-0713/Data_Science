from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def hello_world():
    name = "Dhruvii"
    language = "Python"
    lucky_numbers = [3, 4, 12, 36, 48, 96, 108]
    footer = "<p> Copyright &copy; Dhruvii</p> | All rights reserved"
    return render_template("index.html", name=name, lang=language, lucky=lucky_numbers, footer=footer)

app.run(debug=True)