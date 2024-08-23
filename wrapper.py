from flask import Flask, render_template, request
import cotton

app = Flask(__name__)
bot_mode = 1                # 0 - train, 1 - generate
gen_mode = 1                # 0 - user, 1 - sample
r_author = "shakespeare"

@app.route("/", methods = ["POST", "GET"])
def home():
    return render_template("webui.html")

@app.route("/run_cotton", methods = ["POST", "GET"])
def params():
    global bot_mode, gen_mode, r_author

    b = request.form.get("in_bot_mode")
    g = request.form.get("in_gen_mode")
    r = request.form.get("in_r_author")

    # TODO: error response
    if not b == None and b.isnumeric(): bot_mode = int(b)
    if not g == None and g.isnumeric(): gen_mode = int(g)
    if not r == None and not r == "": r_author = r
    
    print("launching with parameters: " + \
        str(bot_mode) + " " + str(gen_mode) + " " + r_author)

    cotton.set_params(bot_mode, gen_mode, r_author)
    cotton.init_client(cotton.client)
    return "ok"

@app.route("/close_cotton", methods = ["POST", "GET"])
def sleep():
    # TODO: nap not working
    cotton.nap()
    return "close"

if __name__ == "__main__":
    app.run()

