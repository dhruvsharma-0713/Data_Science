from flask import Flask, render_template, request

app = Flask(__name__)

app.secret_key = "Dhruvii's_Secret_Key"

@app.route("/")
def hello_world():
    name = request.args.get("name")
    lang = request.args.get("lang")
    print(name, lang)

    return render_template("index.html", name=name, lang=lang)

app.run(debug=True)