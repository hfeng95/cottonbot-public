#=========================================================
#   cottonbot
#
# discord chat bot that uses gpt2 model to train
# and generate output messages. can be trained to
# parrot user messages or sample texts
#=========================================================

# TODO: 
#   - implement command line args, esp for training
#   - fix formatting of output
#   - update to new discord command format
#   - migrate to aitextgen (partial)
#   - add generation from prompt (partial, auto reply)
#   - merge speak and recite

import discord
import os
import json
import random
import asyncio
import datetime

# from aitextgen
from aitextgen.TokenDataset import TokenDataset
from aitextgen.tokenizers import train_tokenizer
from aitextgen.utils import build_gpt2_config
from aitextgen import aitextgen


BOT_TOKEN = ""
BOT_PREFIX = ""
T_LIMIT = 4000       # limit of messages to be trained on
C_LIMIT = 5          # limit of messages to use for context
L_LIMIT = 512        # max length of generated message
N_STEPS = 100        # steps for finetuning
GPT_MODEL = "gpt2"   # pretrained model
# GPT_MODEL = "EleutherAI/gpt-neo-125M"   # pretrained model

# TODO: take in command line arguments for following values
BOT_MODE = 'generate'       # train, generate, auto
GEN_MODE = 'recite'         # speak (as user), recite (from sample)
R_AUTHOR = "nykko"    # author or user ID for generation

T_MIN_REPLY = 24      # number of seconds minimum between replies

intents = discord.Intents.all()
client = discord.Client(intents = intents)
loop = None
sess = None
cot_ai = None

t_last = None

# write message history of channel to file
async def transcribe(guild, channel):

    print("transcribing on channel " + str(channel.id))
    file_dir = os.path.join("data", str(guild.id))
    file_path = os.path.join(file_dir, str(channel.id) + ".json")

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    # read last T_LIMIT messages in channel
    # export data to json, appending existing file
    with open(file_path, "w+") as o_file:
        print("opening file for transcription")
        msg_list = channel.history(limit = T_LIMIT)
        
        if len(o_file.readlines()) < 1:
            # if file empty, just write everything up to T_LIMIT
            msg_data = []
            async for msg in msg_list:
                if not msg.content.lower().startswith(BOT_PREFIX):
                    msg_data.append({
                        "msg_id": msg.id,
                        "user_id": msg.author.id,
                        "msg": msg.content})
            json.dump(msg_data, o_file, indent = 4)
        
        else:
            # if file not empty, end writing when last entry is reached
            last_entry = json.loads(o_file.readlines()[-1])
            msg_data = []

            async for msg in msg_list:
                if not msg.id == last_entry.msg_id:
                    if not msg.content.lower().startswith(BOT_PREFIX):
                        msg_data.append({
                            "msg_id": msg.id,
                            "user_id": msg.author.id,
                            "msg": msg.content})
                else: break
            json.dump(msg_data, o_file, indent = 4)

        print("learning done")

# train gpt2 on transcribed user on given channel
async def train(ai, guild, channel, user):
    print("training on channel " + str(channel.id))
    file_dir = os.path.join("data", str(guild.id))
    file_path = os.path.join(file_dir, str(channel.id) + ".json")

    # process file generated from transcription
    if os.path.exists(file_path):

        print("processing file for training")
        msg_concat = ""

        with open(file_path, "r") as data_file:
            data = json.load(data_file)
            print("looking for messages by user " + str(user.id))
            for msg_data in data:
                # only process messages from the specified user
                if msg_data["user_id"] == user.id:
                    msg_concat += msg_data["msg"] + "\n"
        
        o_file_path = os.path.join(file_dir, 
            str(channel.id) + "_" + str(user.id) + ".txt")
            
        # TODO: append instead of overwrite?
        with open(o_file_path, "w+") as o_file:
            o_file.write(msg_concat.encode("utf-8").decode("ascii", "ignore"))

            print("finetuning data: " + msg_concat)
            ai.train(
                train_data = TokenDataset(
                    o_file_path,
                    tokenizer_file=os.path.join(
                        "checkpoint",R_AUTHOR,R_AUTHOR+".tokenizer.json"),
                    block_size=L_LIMIT), 
                output_dir = os.path.join("checkpoint",str(user.id)), 
                num_steps = N_STEPS, 
                save_every = int(N_STEPS/20)
            )

        # reset session
        await nap()
        
    else:
        print("must have something to train on, use learn first")
        return
    return

# parrot user based on trained model
# run_name = str(user.id)
async def generate(ai, client, channel, user_id, num_rows):

    # get user by id
    user = await client.fetch_user(user_id)

    if os.path.exists(
        os.path.join("checkpoint", str(user.id))):
        
        # pull context from recent message history,
        # prefix generated from random message from 
        # list of messages
        context = ""
        context_list = channel.history(limit = num_rows)

        async for msg in context_list:
            if not msg.content.startswith(BOT_PREFIX) \
                and not msg.content.startswith("http"):
                context += msg.content

        prefix = random.choice(
            context.split())\
                .replace("```", "\n")\
                .rstrip()\
                .upper()

        print("using prefix: " + prefix)

        await channel.send(
            str(user.name) + " says: ```" + "".join(ai.generate( 
                prompt = prefix, 
                max_length = 50, 
                temperature = 0.85,
                return_as_list = True))\
                    .replace("```", " ")\
                    .replace("<|endoftext|>", " ")\
                    .rsplit('\n', 1)[0] + "```")
    else:
        await channel.send("caw! must train before speak!")

# train on sample text
async def memorize(ai, name):
    print("finetuning data from " + name)
    ai.train(
        train_data = os.path.join("sample", name + ".txt"), 
        output_dir = os.path.join("checkpoint",name), 
        num_steps = N_STEPS, 
        save_every = int(N_STEPS/20)
    )

    print("finetuning over")

    # reset session
    await nap()

# generate from model trained on sample text
# run_name = AUTHOR_NAME
async def recite(ai, channel, text, length, num_rows):

    if os.path.exists(
        os.path.join("checkpoint",text,"pytorch_model.bin")):

        # pull context from recent message history,
        # prefix generated from random message from 
        # list of messages
        context = ""
        context_list = channel.history(limit = num_rows)
        
        async for msg in context_list:
            if not msg.content.startswith(BOT_PREFIX) \
                and not msg.content.startswith("http"):
                context += msg.content

        prefix = random.choice(context.split())\
            .replace("```", "\n")\
            .rstrip()\
            .upper()

        print("reciting using context: " + prefix)

        await channel.send(
            text + " says: ```" + "".join(ai.generate(
                prompt = prefix, 
                max_length = length, 
                temperature = 0.85,
                return_as_list = True))\
                    .replace("<|endoftext|>", " ")\
                    .replace("```", " ")\
                    .rsplit('\n', 1)[0] + "```")
    else:
        await channel.send("caw! must train before speak!")

# read user input
# TODO: error messages for invalid input
# TODO: make train command accessible only to admin
async def process_command(ai, client, guild, channel, command):

    print("processing command on channel " + str(channel.id))
    command_args = command.split()
    if len(command_args) == 0: return   # empty

    # exit command
    if command_args[0].lower() == "leave":
        print("bot exiting")
        await nap()

    # transcribe command
    # BOT_PREFIX learn CHANNEL_INDEX
    if command_args[0].lower() == "learn":
        print("bot learning")

        # check valid channel specified
        if len(command_args) > 1 and command_args[1].isnumeric():
            ch_ind = int(command_args[1])
            if ch_ind >= 0 and ch_ind < len(guild.text_channels):
                await transcribe(guild, \
                    guild.text_channels[ch_ind])
            else: await transcribe(guild, channel)
        else:
            # if no valid channel specified, learn from current channel
            await transcribe(guild, channel)

    # train
    if BOT_MODE == 'train':
        # train command
        # BOT_PREFIX train USER_NAME CHANNEL_INDEX
        if command_args[0].lower() == "train":
            print("bot training")
            if not len(command_args) > 1:
                # no name specified
                await channel.send("caw caw! name pls")
            else:
                user_name = command_args[1]
                user_found = False
                for u in guild.members:
                    if u.name.lower() == user_name.lower():
                        user = u
                        user_found = True

                if user_found == True:
                    # check valid channel specified
                    if len(command_args) > 2 and command_args[2].isnumeric():
                        ch_ind = int(command_args[2])
                        if ch_ind >= 0 and ch_ind < len(guild.text_channels):
                            await train(
                                ai=ai,
                                guild=guild, 
                                channel=guild.text_channels[ch_ind], 
                                user=user)
                        else: await train(
                                ai=ai,
                                guild=guild, 
                                channel=channel, 
                                user=user)
                    else:
                        # if no valid channel specified, 
                        # train on current channel
                        await train(
                            ai=ai,
                            guild=guild, 
                            channel=channel, 
                            user=user)
                else: await channel.send("wrong name, try again!")

        # memorize command
        # BOT_PREFIX memorize AUTHOR_NAME
        elif command_args[0].lower() == "memorize":
            print("bot memorizing")
            if len(command_args) > 1:
                await memorize(ai, command_args[1].lower())
            else:
                await channel.send("need name! cAW!")

    # generate
    elif BOT_MODE == 'generate':
        # generate command
        if command_args[0].lower() == "speak":
            # parrot user
            # BOT_PREFIX speak
            if GEN_MODE == 'speak':
                print("bot speaking")
                async with channel.typing():
                    await generate(
                        ai=ai,
                        client=client, 
                        channel=channel, 
                        user=R_AUTHOR,
                        num_rows=C_LIMIT)

            # recite from sample
            # BOT_PREFIX recite LENGTH
            elif GEN_MODE == 'recite':
                print("bot reciting")

                length = 100

                if len(command_args) > 1:
                    if command_args[1].isnumeric():
                        i = int(command_args[1])
                        length = i
                        if i < 30:
                            length = 30
                        elif i > L_LIMIT:
                            length = L_LIMIT
                async with channel.typing():
                    await recite(
                        ai=ai,
                        channel=channel, 
                        text=R_AUTHOR, 
                        length=length,
                        num_rows=C_LIMIT)

# auto response
async def process_chat(ai, client, channel, user, num_rows):

    print("bot replying")

    # imitate user
    if GEN_MODE == 'speak':
        async with channel.typing():
            await generate(ai, client, channel, R_AUTHOR, num_rows)

    # recite from text
    elif GEN_MODE == 'recite':
        async with channel.typing():
            await recite(ai, channel, R_AUTHOR, 100, num_rows)

async def nap():
    # TODO: doesn't work
    print("taking nap")
    await client.close()

def cry():
    # TODO: cry json
    return

@client.event
async def on_ready():
    print("initiating...")
    init_gpt2()

    # learn
    if BOT_MODE == 'train': activity = discord.Activity(
        name = "and learnin", 
        type = discord.ActivityType.watching)

    # generate
    elif BOT_MODE == 'generate':
        # speak as user
        if GEN_MODE == 'speak': activity = discord.Activity(
            name = "suffering", 
            type = discord.ActivityType.watching)
        # recite from text
        elif GEN_MODE == 'recite': activity = discord.Activity(
            name = R_AUTHOR, 
            type = discord.ActivityType.watching)
    # auto
    elif BOT_MODE == 'auto': activity = discord.Activity(
        name = R_AUTHOR, 
        type = discord.ActivityType.watching)

    await client.change_presence(status = discord.Status.online, \
        activity = activity)

    print("bot ready")

@client.event
async def on_message(message):

    global t_last

    cont = message.content
    user = message.author
    print("message received: " + cont)

    # check that client is running
    if client == None: return

    # check that bot is not referencing own messages
    if user == client.user: return

    # take command
    if BOT_MODE == 'train' or BOT_MODE == 'generate':
        # bot should ignore messages not directed to it
        if not cont.lower().startswith(BOT_PREFIX): return
        
        print("command received: " + cont[len(BOT_PREFIX):])
        await process_command(
            ai=cot_ai, 
            client=client, 
            guild=message.guild, 
            channel=message.channel,
            command=cont[len(BOT_PREFIX):])

    # auto generate in response to messages
    # TODO: should still read certain commands, such as leave
    elif BOT_MODE == 'auto':
        print("time of message: ", message.created_at)

        # make sure enough time has elapsed
        if t_last is None or message.created_at - t_last > \
            datetime.timedelta(seconds = T_MIN_REPLY):
            t_last = message.created_at
            await process_chat(cot_ai, client, message.channel, user, 1)
        else: print("message cooldown")

def set_params(bot, gen, author):
    global BOT_MODE, GEN_MODE, R_AUTHOR

    BOT_MODE = bot
    GEN_MODE = gen
    R_AUTHOR = author

def init_client(client):
    if client is None:
        intents = discord.Intents.all()
        client = discord.Client(intents = intents)

    # TODO: run only on first init
    asyncio.run(client.start(BOT_TOKEN))

def init_gpt2():
    global cot_ai, intents

    # saved model directory
    save_path = os.path.join("checkpoint", R_AUTHOR)

    # if auto reply mode
    if BOT_MODE == 'auto':
        print("initializing cottonbot in auto reply mode")
        cot_ai = aitextgen(
            model_folder=save_path
        )

    # if generate mode
    elif BOT_MODE == 'generate':
        print("initializing cottonbot in generate mode")
        cot_ai = aitextgen(
            model_folder=save_path
        )

    # else if train mode
    elif BOT_MODE == 'train':
        print("initializing cottonbot in training mode")

        # if resuming from existing model
        if os.path.exists(os.path.join(save_path,"pytorch_model.bin")):
            print("existing model found")
            cot_ai = aitextgen(
                model_folder=save_path
            )

        # otherwise download specified model and finetune
        else:     
            print("fetching model")
            cot_ai = aitextgen(
                model=GPT_MODEL
            )

    print("gpt2 loaded")

if __name__ == "__main__":
    init_client(client)