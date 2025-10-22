from flask import Flask, render_template, request, jsonify
import cotton

app = Flask(__name__)

# Default state
settings = {
    "bot_mode": 1,   # 0 - train, 1 - generate
    "gen_mode": 1,   # 0 - user, 1 - sample
    "r_author": "shakespeare",
    "is_running": False
}

@app.route("/", methods=["GET"])
def home():
    return render_template("webui.html", settings=settings)

@app.route("/run_cotton", methods=["POST"])
def run_cotton():
    data = request.form or request.json or {}
    try:
        settings["bot_mode"] = int(data.get("in_bot_mode", settings["bot_mode"]))
        settings["gen_mode"] = int(data.get("in_gen_mode", settings["gen_mode"]))
        r_author = data.get("in_r_author", "").strip()
        if r_author:
            settings["r_author"] = r_author
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid parameter types"}), 400

    cotton.set_params(settings["bot_mode"], settings["gen_mode"], settings["r_author"])
    cotton.init_client(cotton.client)
    settings["is_running"] = True

    print(f"Launching with parameters: {settings}")
    return jsonify({"status": "ok", "settings": settings})

@app.route("/close_cotton", methods=["POST"])
def close_cotton():
    cotton.nap()
    settings["is_running"] = False
    return jsonify({"status": "closed"})

@app.route("/status", methods=["GET"])
def status():
    """Optional: let the frontend query current bot state."""
    return jsonify(settings)

if __name__ == "__main__":
    app.run(debug=True)
