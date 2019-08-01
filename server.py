from flask import Flask, request, render_template, url_for
import type_predictor

user_data = []

app = Flask(__name__)

@app.route("/main")
def home():
    return render_template("index.html")

@app.route("/test")
def test():
    return render_template("form.html")

@app.route("/get_data",methods=["POST"])
def get_data():
    global user_data
    for i in range(5):
        answer = request.form["ans"+str(i+1)]
        user_data.append( answer )
    user_type = type_predictor.prediction(user_data)
    name = request.form["name"]
    return render_template("response.html", name=name, user_type=user_type)

if __name__ == "__main__":
    app.run(debug=True)
