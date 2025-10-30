from flask import Flask, render_template, request

app = Flask(__name__, static_folder="assets", template_folder="templates")

@app.route("/", methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        print(f"The email is: {email} and the password is: {password}")
        return "<b>Thanks for submitting the form.</b>"
    return render_template("index.html")

app.run(debug=True)