from flask import Flask, render_template, request, jsonify
import subprocess
import signal
import os

app = Flask(__name__)

settings = {
    "bot_mode": 1,    # 0 - train, 1 - generate
    "gen_mode": 1,    # 0 - user, 1 - sample
    "r_author": "shakespeare",
    "is_running": False,
}

bot_process = None  # Store the bot subprocess


@app.route("/", methods=["GET"])
def home():
    return render_template("webui.html", settings=settings)


@app.route("/run_cotton", methods=["POST"])
def run_cotton():
    global bot_process
    data = request.form or request.json or {}
    try:
        settings["bot_mode"] = int(data.get("in_bot_mode", settings["bot_mode"]))
        settings["gen_mode"] = int(data.get("in_gen_mode", settings["gen_mode"]))
        r_author = data.get("in_r_author", "").strip()
        if r_author:
            settings["r_author"] = r_author
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid parameter types"}), 400

    if bot_process and bot_process.poll() is None:
        return jsonify({"status": "error", "message": "Bot already running"}), 400

    # Launch cotton.py with parameters as CLI arguments
    cmd = [
        "python",
        "cotton.py",
        str(settings["bot_mode"]),
        str(settings["gen_mode"]),
        settings["r_author"],
    ]

    bot_process = subprocess.Popen(
        cmd,
        text=True,
    )

    settings["is_running"] = True
    print(f"Launching CottonBot with parameters: {settings}")
    return jsonify({"status": "ok", "settings": settings})


@app.route("/close_cotton", methods=["POST"])
def close_cotton():
    global bot_process

    if bot_process and bot_process.poll() is None:
        print("Terminating CottonBot subprocess...")
        bot_process.terminate()

        try:
            bot_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("Bot unresponsive — force killing.")
            bot_process.kill()

        bot_process = None
        settings["is_running"] = False
        return jsonify({"status": "closed"})
    else:
        return jsonify({"status": "error", "message": "Bot is not running"}), 400


@app.route("/status", methods=["GET"])
def status():
    return jsonify(settings)


if __name__ == "__main__":
    app.run(debug=True)
