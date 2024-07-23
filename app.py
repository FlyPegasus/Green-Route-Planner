from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home_view():
    return render_template("home.html")

@app.route("/map")
def map_view():
    return render_template("map.html")


if __name__=="__main__":
    app.run(debug=True)