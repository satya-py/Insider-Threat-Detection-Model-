from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
def index():
    return jsonify({"message": "Hello from Flask on Vercel!"})

# If you already have a Blueprint or an `app` object in another file,
# import it here and expose it as `app`.
#
# e.g. from myapp import create_app
# app = create_app()

if __name__ == "__main__":
    app.run()
