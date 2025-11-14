# cottonbot

## Installation
```
git clone https://github.com/hfeng95/cottonbot-public.git
cd cottonbot-public
conda env create --file environment.yml
```

You need a Discord bot token to host cottonbot. You can find information
about that [here](https://discord.com/developers/docs/intro).

## Usage
Activate the conda environment with:
```
conda activate cottonbot
```

The core functionalities of cottonbot are user/channel-specific,
so cottonbot must be present on the server where the user/channel
is active. Make sure to set `BOT_TOKEN` with your Discord bot token and
`PRIV_ID` with channel ID if you want cottonbot to run only on a specified
channel.

For TTS features, you need FFmpeg. See [here](https://www.ffmpeg.org/)
for more info. Place `ffmpeg.exe` in the main folder.

### Training
To extract data from the message history of a Discord channel, 
cottonbot must have access to the channel.

Use `cotton_train.py` to fine-tune your custom model. `train.bat`
shows an example of launch arguments.

### Running
To run cottonbot, run `launch_cotton_web.bat`. This will install Flask
if it hasn't been installed already and start the Web UI which can be 
accessed via `http://127.0.0.1:5000/`.

From here, you can:
- Choose bot mode:
  - Learning: for gathering training data on channel messages. Use the 
`learn` bot command. Use responsibly with the consent of all affected parties.
  - Speaking: for inference.
- Specify behavior and author. Use the auto-reply behavior for agentic functions.
  - For OpenAI, put the model as the author. For example, "gpt-5".

Click on Awaken/Slumber to start/close the bot.

Alternatively, you can run `cotton.py` directly.