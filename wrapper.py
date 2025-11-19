from flask import Flask, render_template, request, jsonify
import subprocess
import signal
import os

app = Flask(__name__)

settings = {
    "bot_mode": 'speak',
    "gen_mode": 'command',
    "r_author": "shakespeare",
    "adaptive": True,
    "temperature": 0.7,
    "tts_enabled": False,
    "modder_enabled": True,
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
        bot_mode = data.get("in_bot_mode", "").strip()
        if bot_mode:
            settings["bot_mode"] = bot_mode
        gen_mode = data.get("in_gen_mode", "").strip()
        if gen_mode:
            settings["gen_mode"] = gen_mode
        r_author = data.get("in_r_author", "").strip()
        if r_author:
            settings["r_author"] = r_author
        
        # Handle boolean/feature settings
        adaptive = data.get("in_adaptive", "").strip().lower()
        if adaptive in ('true', 'false', '1', '0'):
            settings["adaptive"] = adaptive in ('true', '1')
        
        temperature = data.get("in_temperature", "").strip()
        if temperature:
            try:
                settings["temperature"] = float(temperature)
            except ValueError:
                pass
        
        tts_enabled = data.get("in_tts_enabled", "").strip().lower()
        if tts_enabled in ('true', 'false', '1', '0'):
            settings["tts_enabled"] = tts_enabled in ('true', '1')
        
        modder_enabled = data.get("in_modder_enabled", "").strip().lower()
        if modder_enabled in ('true', 'false', '1', '0'):
            settings["modder_enabled"] = modder_enabled in ('true', '1')
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid parameter types"}), 400

    if bot_process and bot_process.poll() is None:
        return jsonify({"status": "error", "message": "Bot already running"}), 400

    # Launch cotton.py with parameters as CLI arguments
    cmd = [
        "python",
        "cotton.py",
        "--mode", settings["bot_mode"],
        "--behavior", settings["gen_mode"],
        "--author", settings["r_author"],
    ]
    
    # Add feature flags if they differ from defaults or are explicitly set
    if settings.get("adaptive") is not None:
        cmd.extend(["--adaptive", str(settings["adaptive"]).lower()])
    if settings.get("temperature") is not None:
        cmd.extend(["--temperature", str(settings["temperature"])])
    if settings.get("tts_enabled") is not None:
        cmd.extend(["--tts-enabled", str(settings["tts_enabled"]).lower()])
    if settings.get("modder_enabled") is not None:
        cmd.extend(["--modder-enabled", str(settings["modder_enabled"]).lower()])

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
