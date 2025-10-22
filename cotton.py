#=========================================================
#   cottonbot (revamped)
#
# Discord chatbot using Hugging Face Transformers
# Supports local fine-tuning and message generation
#=========================================================

<<<<<<< HEAD
# TODO: 
#   - implement command line args, esp for training
#   - fix formatting of output
#   - update to new discord command format
#   - migrate to aitextgen (partial)
#   - add generation from prompt (partial, auto reply)
#   - merge speak and recite

=======
>>>>>>> bb41b28 (Revamp to use huggingface transformers directly)
import discord
import os
import json
import random
import asyncio
import datetime
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TextDataset,
    DataCollatorForLanguageModeling
)

with open('config.json') as f:
    f_data = json.load(f)
    BOT_TOKEN = f_data['BOT_TOKEN']
    PRIV_ID = f_data['PRIV_ID']

BOT_PREFIX = "&cotton "
HISTORY_LIMIT = 4000
CONTEXT_LIMIT = 5
LENGTH_LIMIT = 512
N_STEPS = 100
BASE_MODEL = "LiquidAI/LFM2-350M"

BOT_MODE = 1   # 0 - train, 1 - generate, 2 - auto
GEN_MODE = 1   # 0 - user mimicry, 1 - sample text recitation
PRIV_MODE = True
R_AUTHOR = "nykko"

TIME_MIN_REPLY = 24

intents = discord.Intents.all()
client = discord.Client(intents=intents)

cotton_tokenizer = None
cotton_model = None
time_last_msg = None


# -------------------------------------------------------------
# WebUI
# -------------------------------------------------------------
def set_params(bot, gen, author):
    global BOT_MODE, GEN_MODE, R_AUTHOR

    BOT_MODE = bot
    GEN_MODE = gen
    R_AUTHOR = author

# -------------------------------------------------------------
# Utility: Load/Save and fine-tune the model
# -------------------------------------------------------------
def load_model(author: str):
    """Load fine-tuned model if exists, else base model."""
    save_path = os.path.join("checkpoint", author)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token  # For stability
    if os.path.exists(os.path.join(save_path, "model.safetensors")):
        print(f"Loading fine-tuned model for {author}")
        model = AutoModelForCausalLM.from_pretrained(save_path, use_safetensors=True)
    elif os.path.exists(os.path.join(save_path, "pytorch_model.bin")):
        print(f"Loading fine-tuned model for {author}")
        model = AutoModelForCausalLM.from_pretrained(save_path)
    else:
        print(f"Loading base model: {BASE_MODEL}")
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    if(torch.cuda.is_available()):
        model.to('cuda')
        print('CUDA found.')
    else:
        print('CUDA not found. Using CPU.')
    return tokenizer, model


def fine_tune_model(train_path: str, output_dir: str, steps: int = N_STEPS):
    """Fine-tune GPT model locally using Hugging Face Trainer."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    def load_dataset(path):
        return TextDataset(
            tokenizer=tokenizer,
            file_path=path,
            block_size=128
        )

    train_dataset = load_dataset(train_path)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        save_steps=max(1, steps // 10),
        save_total_limit=2,
        logging_dir="./logs",
        max_steps=steps
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


# -------------------------------------------------------------
# Async functions (same as old bot, but simplified)
# -------------------------------------------------------------
async def transcribe(guild, channel):
    print(f"Transcribing messages from {channel.name}...")
    file_dir = os.path.join("data", str(guild.id))
    os.makedirs(file_dir, exist_ok=True)
    file_path = os.path.join(file_dir, f"{channel.id}.json")

    msg_data = []
    async for msg in channel.history(limit=HISTORY_LIMIT):
        if not msg.content.lower().startswith(BOT_PREFIX):
            msg_data.append({
                "msg_id": msg.id,
                "user_id": msg.author.id,
                "msg": msg.content
            })
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(msg_data, f, indent=2)
    print(f"Transcription complete: {len(msg_data)} messages saved.")


async def find_prompt(channel, num_rows):
    context_list = []
    async for msg in channel.history(limit=num_rows):
        if not msg.content.startswith(BOT_PREFIX) and not msg.content.startswith("http"):
            context_list.append(msg.content)

    if not context_list:
        return "COTTON"
    
    # select random message from context and random word from the message
    msg = random.choice(context_list).strip().replace("```", "\n")
    words = msg.split()
    if words:
        prefix = random.choice(words)
    else:
        prefix = ""
    return prefix


async def train_model(guild, channel, user):
    print(f"Training on channel {channel.name} for {user.name}")
    file_dir = os.path.join("data", str(guild.id))
    file_path = os.path.join(file_dir, f"{channel.id}.json")

    if not os.path.exists(file_path):
        print("No transcription file found.")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    user_msgs = [d["msg"] for d in data if d["user_id"] == user.id]
    if not user_msgs:
        print("No messages found for that user.")
        return

    o_file_path = os.path.join(file_dir, f"{channel.id}_{user.id}.txt")
    with open(o_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(user_msgs))

    fine_tune_model(o_file_path, os.path.join("checkpoint", str(user.id)), N_STEPS)


async def generate_text(channel, prompt, author, max_len=100):
    global cotton_tokenizer, cotton_model
    if cotton_tokenizer is None or cotton_model is None:
        cotton_tokenizer, cotton_model = load_model(author)

    inputs = cotton_tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        outputs = cotton_model.generate(
            **inputs,
            do_sample=True,
            max_length=max_len,
            temperature=0.7,
            repetition_penalty=1.5,
            pad_token_id=cotton_tokenizer.eos_token_id
        )

    text = cotton_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(text)
    await channel.send(f"{author} says: ```{text.strip()}```")


# -------------------------------------------------------------
# Command and Message Handling
# -------------------------------------------------------------
@client.event
async def on_ready():
    global cotton_tokenizer, cotton_model
    cotton_tokenizer, cotton_model = load_model(R_AUTHOR)

    activity_name = {
        0: "and learnin",
        1: "suffering" if GEN_MODE == 0 else R_AUTHOR,
        2: R_AUTHOR
    }[BOT_MODE]
    activity = discord.Activity(
        name=activity_name, type=discord.ActivityType.watching
    )
    await client.change_presence(status=discord.Status.online, activity=activity)
    print(f"Cottonbot is ready.")


@client.event
async def on_message(message):
    global time_last_msg

    if message.author == client.user:
        return

    content = message.content.strip()
    channel = message.channel
    guild = message.guild

    print(f'Message received: \"{content}\" from guild {guild} ({guild.id}) channel {channel} ({channel.id})')

    if PRIV_MODE and not channel.id == PRIV_ID:
        return

    if BOT_MODE in (0, 1) and content.lower().startswith(BOT_PREFIX):
        command = content[len(BOT_PREFIX):].split()
        if not command:
            return

        cmd = command[0].lower()

        print('Command read:',cmd)

        if cmd == "learn":
            await transcribe(guild, channel)
        elif cmd == "train" and len(command) > 1:
            user_name = command[1]
            user = discord.utils.find(lambda m: m.name.lower() == user_name.lower(), guild.members)
            if user:
                await train_model(guild, channel, user)
            else:
                await channel.send("User not found!")
        elif cmd == "speak":
            prefix = await find_prompt(channel, CONTEXT_LIMIT)
            try:
                max_len = int(command[1])
            except (IndexError, ValueError):
                max_len = LENGTH_LIMIT
            async with channel.typing():
                await generate_text(channel, prefix, R_AUTHOR, max_len=max_len)

    elif BOT_MODE == 2:
        if time_last_msg is None or message.created_at - time_last_msg > datetime.timedelta(seconds=TIME_MIN_REPLY):
            time_last_msg = message.created_at
            prefix = await find_prompt(channel, 1)
            await generate_text(channel, prefix, R_AUTHOR, max_len=100)
        else:
            print("Message cooldown active.")


if __name__ == "__main__":
    asyncio.run(client.start(BOT_TOKEN))
