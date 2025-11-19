#=========================================================
#   cottonbot (revamped)
#
# Discord chatbot using Hugging Face Transformers
# Supports local fine-tuning and message generation
#=========================================================

import discord
import os
import json
import random
import asyncio
import datetime
import torch
import sys
import argparse
import torch
import torch.nn.functional as F
import soundfile as sf
import nacl
from memory_manager import CottonMemory
from wikier import WikiAgent
from modder import ModAgent
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TextDataset,
    DataCollatorForLanguageModeling,
    pipeline
)

with open('config.json') as f:
    f_data = json.load(f)
    BOT_TOKEN = f_data['BOT_TOKEN']
    PRIV_ID = f_data['PRIV_ID']

# TODO: CLI for all parameters
BOT_PREFIX = "$cotton "
HISTORY_LIMIT = 4000
CONTEXT_LIMIT = 5
LENGTH_LIMIT = 512  # TODO: implement frontend
N_STEPS = 100
BASE_MODEL = "LiquidAI/LFM2-350M"
MEMORY_MODEL = "LiquidAI/LFM2-350M"
TTS_MODEL = "facebook/mms-tts-eng"
MODDER_MODEL = "s-nlp/roberta_toxicity_classifier"
MODDER_CAUSAL_MODEL = None

BOT_MODE = 'speak'   # listen/speak
GEN_MODE = 'auto'   # command/auto
PRIV_MODE = True
R_AUTHOR = "nykko"
ADAPTIVE = True
TEMPERATURE = 0.7
TTS_ENABLED = False  # TODO: implement command toggle per server
MODDER_ENABLED = True


TIME_MIN_REPLY = 8

intents = discord.Intents.all()
client = discord.Client(intents=intents)

cotton_tokenizer = None
cotton_model = None
router_pipeline = None
wiki_agent = None
tts_tokenizer = None
tts_model = None
openai_client = None
modder_agent = None

time_last_msg = None    # global variable for now. TODO: separate for each server/channel

client_loop_ref = None

memory_manager_dict = {}

SERVER_SETTINGS_FILE = "server_settings.json"

def load_server_settings():
    """Load server-specific settings from JSON file."""
    if os.path.exists(SERVER_SETTINGS_FILE):
        try:
            with open(SERVER_SETTINGS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_server_settings(server_settings):
    """Save server-specific settings to JSON file."""
    with open(SERVER_SETTINGS_FILE, 'w') as f:
        json.dump(server_settings, f, indent=2)

def get_server_setting(guild_id, setting_name, default_value):
    """Get a server-specific setting, with fallback to global default."""
    server_settings = load_server_settings()
    guild_id_str = str(guild_id)
    
    if guild_id_str not in server_settings:
        return default_value
    
    return server_settings[guild_id_str].get(setting_name, default_value)


# -------------------------------------------------------------
# Utility
# -------------------------------------------------------------
def set_params(bot, gen, author):
    global BOT_MODE, GEN_MODE, R_AUTHOR

    BOT_MODE = bot
    GEN_MODE = gen
    R_AUTHOR = author

def set_model_params(base_model=None, memory_model=None, tts_model=None, modder_model=None, modder_causal_model=None):
    global BASE_MODEL, MEMORY_MODEL, TTS_MODEL, MODDER_MODEL, MODDER_CAUSAL_MODEL
    
    if base_model:
        BASE_MODEL = base_model
    if memory_model:
        MEMORY_MODEL = memory_model
    if tts_model:
        TTS_MODEL = tts_model
    if modder_model:
        MODDER_MODEL = modder_model
    if modder_causal_model:
        MODDER_CAUSAL_MODEL = modder_causal_model

def set_feature_params(adaptive=None, temperature=None, tts_enabled=None, modder_enabled=None):
    global ADAPTIVE, TEMPERATURE, TTS_ENABLED, MODDER_ENABLED
    
    if adaptive is not None:
        ADAPTIVE = adaptive
    if temperature is not None:
        TEMPERATURE = temperature
    if tts_enabled is not None:
        TTS_ENABLED = tts_enabled
    if modder_enabled is not None:
        MODDER_ENABLED = modder_enabled

def parse_args():
    parser = argparse.ArgumentParser(description="CottonBot configuration")
    parser.add_argument("--mode", type=str, default='speak', help="listen/speak")
    parser.add_argument("--behavior", type=str, default='command', help="command/auto")
    parser.add_argument("--author", type=str, default="shakespeare", help="target author name")
    parser.add_argument("--base-model", type=str, default=None, help="Base model for text generation (default: LiquidAI/LFM2-350M)")
    parser.add_argument("--memory-model", type=str, default=None, help="Model for memory management (default: LiquidAI/LFM2-350M)")
    parser.add_argument("--tts-model", type=str, default=None, help="TTS model for speech synthesis (default: facebook/mms-tts-eng)")
    parser.add_argument("--modder-model", type=str, default=None, help="Toxicity classifier model (default: s-nlp/roberta_toxicity_classifier)")
    parser.add_argument("--modder-causal-model", type=str, default=None, help="Causal model for moderation reasoning (default: meta-llama/Llama-3.1-8B-Instruct)")
    def str_to_bool(v):
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser.add_argument("--adaptive", type=str_to_bool, default=None, help="Enable adaptive text generation (true/false)")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature for text generation (0.0-2.0, default: 0.7)")
    parser.add_argument("--tts-enabled", type=str_to_bool, default=None, help="Enable TTS features (true/false)")
    parser.add_argument("--modder-enabled", type=str_to_bool, default=None, help="Enable moderation features (true/false)")
    return parser.parse_args()

# -------------------------------------------------------------
# Model functions: Load/Save and fine-tune the model
# -------------------------------------------------------------
def load_model(author: str):
    """Load fine-tuned model if exists, else base model."""
    if 'gpt' in author:
        print(f'GPT model name detected, loading OpenAI client.')
        from openai import OpenAI
        global openai_client
        openai_client = OpenAI()     # TODO: needs to be tested
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

def load_router():
    print('Loading tool router')
    router = pipeline(
        "zero-shot-classification",
        model="sileod/deberta-v3-xsmall-tasksource-nli",
        device_map="auto"
    )
    return router

def load_tts_model(model_name=None):
    if model_name is None:
        model_name = TTS_MODEL
    print(f'Loading TTS model: {model_name}')
    from transformers import VitsModel
    model = VitsModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if(torch.cuda.is_available()):
        # model.to('cuda') TODO: not working
        print('TTS model loaded with CUDA.')
    return tokenizer,model

# DEPRECATED: training should be done in cotton_train.py
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
    print(f"Transcription complete: {len(msg_data)} messages saved to {file_path}.")


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


async def extract_user_msgs(guild, channel, user):
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

    print(f'Messages from user {user.name} saved to {o_file_path}.')


async def consult_agent(query: str):
    """Return the most likely category of a user query."""
    """Selects appropriate agent/tool and returns output"""
    labels = ["history", "politics", "science", "geography", "programming", "entertainment", "storytelling", "casual conversation"]
    result = router_pipeline(query, candidate_labels=labels)
    category = result["labels"][0]                # Top prediction
    print(f"[Router] Category: {category}")

    if category in ["history", "politics", "science", "geography"]:
        # call wiki agent
        wiki_results = wiki_agent.lookup_body(query)
        if not wiki_results: return None
        return wiki_results[0]

    # TODO: coding agent

    return None

async def toxicity_filter(text: str):
    is_toxic = await modder_agent.check_toxicity(text)
    return is_toxic

async def parse_moderation_action(judge_response: str):
    """
    Parse the judge response to extract moderation action.
    Returns: (action, reason) where action is one of: warn, delete, timeout, kick, ban
    """
    response_lower = judge_response.lower()
    
    # Extract action from response
    if 'ban' in response_lower:
        return ('ban', judge_response)
    elif 'kick' in response_lower:
        return ('kick', judge_response)
    elif 'timeout' in response_lower:
        return ('timeout', judge_response)
    elif 'delete' in response_lower or 'deleted' in response_lower:
        return ('delete', judge_response)
    elif 'warn' in response_lower or 'warning' in response_lower:
        return ('warn', judge_response)
    else:
        # Default to delete if no clear action found
        return ('delete', judge_response)

async def execute_moderation_action(message, action: str, reason: str):
    """
    Execute the moderation action on the message/author.
    For kick/ban, only timeout and recommend the action.
    """
    author = message.author
    channel = message.channel
    guild = message.guild
    
    try:
        if action == 'delete':
            await message.delete()
            await channel.send(f"Message deleted. Reason: {reason[:200]}")
            return True
        
        elif action == 'warn':
            await channel.send(f"⚠️ Warning to {author.mention}: {reason[:200]}")
            return True
        
        elif action == 'timeout':
            # Timeout for 1 hour
            timeout_until = datetime.datetime.utcnow() + datetime.timedelta(hours=1)
            try:
                # message.author is already a Member object in guild context
                if isinstance(author, discord.Member):
                    await author.timeout(timeout_until, reason=reason[:200])
                    await channel.send(f"⏱️ {author.mention} has been timed out for 1 hour. Reason: {reason[:200]}")
                    return True
                else:
                    await channel.send(f"❌ Cannot timeout {author.mention}: User not found in server.")
                    return False
            except discord.Forbidden:
                await channel.send(f"❌ Cannot timeout {author.mention}: Insufficient permissions.")
                return False
            except Exception as e:
                print(f"Error timing out user: {e}")
                await channel.send(f"❌ Error timing out {author.mention}: {str(e)}")
                return False
        
        elif action == 'kick':
            # Timeout instead and recommend kick
            timeout_until = datetime.datetime.utcnow() + datetime.timedelta(hours=1)
            try:
                if isinstance(author, discord.Member):
                    await author.timeout(timeout_until, reason=f"Recommended kick: {reason[:200]}")
                    await channel.send(
                        f"⏱️ {author.mention} has been timed out. "
                        f"**Recommendation:** Consider kicking this user. Reason: {reason[:200]}"
                    )
                    return True
                else:
                    await channel.send(f"❌ Cannot timeout {author.mention}: User not found in server.")
                    return False
            except discord.Forbidden:
                await channel.send(f"❌ Cannot timeout {author.mention}: Insufficient permissions.")
                return False
            except Exception as e:
                print(f"Error timing out user (kick recommendation): {e}")
                await channel.send(f"❌ Error timing out {author.mention}: {str(e)}")
                return False
        
        elif action == 'ban':
            # Timeout instead and recommend ban
            timeout_until = datetime.datetime.utcnow() + datetime.timedelta(hours=24)
            try:
                if isinstance(author, discord.Member):
                    await author.timeout(timeout_until, reason=f"Recommended ban: {reason[:200]}")
                    await channel.send(
                        f"⏱️ {author.mention} has been timed out for 24 hours. "
                        f"**Recommendation:** Consider banning this user. Reason: {reason[:200]}"
                    )
                    return True
                else:
                    await channel.send(f"❌ Cannot timeout {author.mention}: User not found in server.")
                    return False
            except discord.Forbidden:
                await channel.send(f"❌ Cannot timeout {author.mention}: Insufficient permissions.")
                return False
            except Exception as e:
                print(f"Error timing out user (ban recommendation): {e}")
                await channel.send(f"❌ Error timing out {author.mention}: {str(e)}")
                return False
        
        return False
    except Exception as e:
        print(f"Error executing moderation action {action}: {e}")
        return False

async def generate_text(channel, prompt, author, max_len=100, include_prefix=True, adaptive=False):
    global cotton_tokenizer, cotton_model
    if cotton_tokenizer is None or cotton_model is None:
        cotton_tokenizer, cotton_model = load_model(author)

    if adaptive:
        text = await generate_adaptive(
            model=cotton_model,
            tokenizer=cotton_tokenizer,
            inputs=prompt,
            max_new_tokens=max_len,
            temperature=TEMPERATURE,
            repetition_penalty=1.2,
            entropy_threshold=4.0,
            patience=5,
            max_sentences=2
        )
        print('Output (with adaptive termination):',text)
        await channel.send(f"{author} says: ```{text.strip()}```")
        return text

    # if prompt is a string, encode. otherwise, assume it is a dict of tokenized ids.
    if isinstance(prompt,str):
        inputs = cotton_tokenizer(prompt, return_tensors="pt")
    else:
        inputs = prompt

    if include_prefix==True:    # include prefix in output
        with torch.no_grad():
            outputs = cotton_model.generate(
                **inputs,
                do_sample=True,
                max_length=max_len,
                temperature=TEMPERATURE,
                repetition_penalty=1.2,
                pad_token_id=cotton_tokenizer.eos_token_id,
                eos_token_id=cotton_tokenizer.eos_token_id
            )

        text = cotton_tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        with torch.no_grad():
            outputs = cotton_model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=max_len,
                temperature=TEMPERATURE,
                repetition_penalty=1.2,
                pad_token_id=cotton_tokenizer.eos_token_id,
                eos_token_id=cotton_tokenizer.eos_token_id
            )
        text = cotton_tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    print('Output:',text)
    await channel.send(f"{author} says: ```{text.strip()}```")
    return text


# experimental, adaptive stopping
# TODO: if we reach max token limit, trim output from last sentence end
async def generate_adaptive(
    model,
    tokenizer,
    inputs,
    max_new_tokens=256,
    temperature=0.7,
    repetition_penalty=1.5,
    entropy_threshold=4.0,
    patience=5,
    max_sentences=None
):
    input_ids = inputs["input_ids"].clone()
    past_key_values = None
    uncertain_steps = 0
    sentence_count = 0

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :] / temperature

            # Apply repetition penalty
            for token_id in set(input_ids[0].tolist()):
                logits[0, token_id] /= repetition_penalty

            probs = F.softmax(logits, dim=-1)

            # Compute entropy (measure of uncertainty)
            entropy = -torch.sum(probs * probs.log(), dim=-1).item()

            # Increment uncertainty counter
            if entropy > entropy_threshold:
                uncertain_steps += 1
            else:
                uncertain_steps = 0

            # Stop if model stays uncertain for too long
            if uncertain_steps >= patience:
                break

            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values

            # check if we exceed max sentence limit
            if max_sentences:
                next_str = tokenizer.decode(next_token[0],skip_special_tokens=True)
                if next_str.strip().endswith(('.', '!', '?')):
                    sentence_count += 1
                if sentence_count > max_sentences:
                    break

            # Stop if EOS token is reached
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode only the newly generated portion
    return tokenizer.decode(
        input_ids[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

async def generate_openai(
    channel,
    openai_client,
    prompt,
    max_new_tokens=256,
    temperature=0.7
):
    response = openai_client.chat.completions.create(
        model=R_AUTHOR,  # Or your chosen model
        messages=prompt,
        max_tokens=max_new_tokens,  # Limits the length of the generated response
        temperature=temperature, # Controls randomness (0.0-1.0)
    )
    text = response.choices[0].message.content
    await channel.send(f"{R_AUTHOR} says: ```{text.strip()}```")
    return text


async def play_tts(author, guild, text):

    if not author.voice:
        print("User is not in a voice channel.")
        return

    # Generate TTS audio
    print('Synthesizing speech')
    inputs = tts_tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        speech = tts_model(**inputs).waveform.to(tts_model.device)

    print('Writing sound file')
    output_path="tts_output.wav"
    try:
        sf.write(output_path, speech.squeeze().cpu().numpy(), 16000)
    except Exception as e:
        print(f'Error: {e}')

    voice_channel = author.voice.channel

    if not guild.voice_client:
        print('Connecting to voice channel',voice_channel)
        vc = await voice_channel.connect()
    else:
        vc = guild.voice_client
        if vc.channel != voice_channel:
            print('Moving to voice channel',voice_channel)
            await vc.move_to(voice_channel)

    # Use ffmpeg to stream to Discord
    try:
        vc.play(discord.FFmpegPCMAudio(executable="ffmpeg.exe", source=output_path))
    except Exception as e:
        print(f'Error: {e}')
    while vc.is_playing():
        await asyncio.sleep(0.5)


async def build_chat_prompt(tokenizer, system_message, user_message, context=None, recent_conversation=None, return_dict=False):
    if context is None:
        context = ''
    else:
        context += ' '
    if recent_conversation: # if multiple rounds of conversation are provided
        messages = [{"role": "system", "content": context+system_message}]
        for user_turn,bot_turn in recent_conversation:
            user_name,user_msg = user_turn
            bot_name,bot_msg = bot_turn
            messages.append({'role':'user','content':user_msg})
            messages.append({'role':'assistant','content':bot_msg})
        messages.append({"role": "user", "content": user_message})
    else:
        messages = [
            {"role": "system", "content": context+system_message},
            {"role": "user", "content": user_message}
        ]
    if return_dict:
        return messages
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(cotton_model.device)


# -------------------------------------------------------------
# Command and Message Handling
# -------------------------------------------------------------
@client.event
async def on_ready():
    global cotton_tokenizer, cotton_model, router_pipeline, wiki_agent, tts_tokenizer, tts_model, openai_client, modder_agent
    if not openai_client:
        cotton_tokenizer, cotton_model = load_model(R_AUTHOR)
    router_pipeline = load_router()
    wiki_agent = WikiAgent()
    # TTS and Modder are loaded globally but enabled per-server
    if TTS_ENABLED:
        tts_tokenizer,tts_model = load_tts_model()
    if MODDER_ENABLED:
        modder_agent = ModAgent(
            toxicity_model_name=MODDER_MODEL,
            causal_model_name=MODDER_CAUSAL_MODEL
        )

    activity_name = {
        'listen': "and learning",
        'speak': R_AUTHOR
    }[BOT_MODE]
    activity = discord.Activity(
        name=activity_name, type=discord.ActivityType.watching
    )
    await client.change_presence(status=discord.Status.online, activity=activity)

    # if auto-reply behavior is selected, initialize memory
    if GEN_MODE == 'auto':
        if PRIV_MODE:
            guild,channel = client.get_channel(PRIV_ID).guild,client.get_channel(PRIV_ID)
            print(f'Loading memory for guild {guild} ({guild.id}) channel {channel} ({channel.id}).')
            memory_manager_dict[guild.id] = {}
            memory_manager_dict[guild.id][channel.id] = CottonMemory(model_type='no-chain')
        else:
            for guild in client.guilds:
                bot_id = guild.get_member(client.user.id)
                memory_manager_dict[guild.id] = {}
                for channel in guild.text_channels:
                    if not channel.permissions_for(bot_id).send_messages:
                        continue
                    print(f'Loading memory for guild {guild} ({guild.id}) channel {channel} ({channel.id}).')
                    memory_manager_dict[guild.id][channel.id] = CottonMemory(model_type='no-chain')

    print(f"Cottonbot is ready.")


@client.event
async def on_message(message):
    global time_last_msg

    # don't react to own messages
    if message.author == client.user:
        return

    content = message.content.strip()
    channel = message.channel
    guild = message.guild

    print(f'Message received: \"{content}\" from guild {guild} ({guild.id}) channel {channel} ({channel.id})')

    if PRIV_MODE and not channel.id == PRIV_ID:
        return
    
    # check toxicity (per-server setting)
    server_modder_enabled = get_server_setting(guild.id, "modder_enabled", MODDER_ENABLED)
    if server_modder_enabled and modder_agent:
        # Check if causal model is available for advanced moderation
        has_causal_model = hasattr(modder_agent, 'llm') and modder_agent.llm is not None
        
        if has_causal_model:
            # Use judge function for advanced moderation. TODO: needs testing
            try:
                judge_response = await modder_agent.judge(content)
                action, reason = await parse_moderation_action(judge_response)
                
                # Check if message should be allowed
                if 'yes' in judge_response.lower()[:50] and 'no' not in judge_response.lower()[:50]:
                    # Message is allowed, continue processing
                    pass
                else:
                    # Message should be moderated
                    await execute_moderation_action(message, action, reason)
                    return
            except Exception as e:
                print(f"Error in moderation judge: {e}")
                # Fallback to simple toxicity check
                toxicity = await toxicity_filter(content)
                if toxicity and toxicity > 0.995:
                    await message.delete()
                    await channel.send(f"Message deleted for toxicity.")
                    return
        else:
            # Fallback to simple toxicity check if no causal model
            toxicity = await toxicity_filter(content)
            if toxicity and toxicity > 0.995:
                await message.delete()
                await channel.send(f"Message deleted for toxicity.")
                return
    
    # learning mode
    if BOT_MODE == 'listen' and content.lower().startswith(BOT_PREFIX):
        command = content[len(BOT_PREFIX):].split()
        if not command:
            return
        cmd = command[0].lower()

        print('Command read:',cmd)

        if cmd == "learn":
            await transcribe(guild, channel)
            if len(command) > 1:
                # if name is specified, extract messages from the user
                user_name = command[1]
                user = discord.utils.find(lambda m: m.name.lower() == user_name.lower(), guild.members)
                if user:
                    await channel.send(f"Collecting data from user {user} on channel {channel}.")
                    await extract_user_msgs(guild, channel, user)
                else:
                    await channel.send(f"User {user} not found on channel {channel}!")

    # speaking mode, command behavior
    elif BOT_MODE == 'speak' and GEN_MODE == 'command':
        if not content.lower().startswith(BOT_PREFIX):
            return
        command = content[len(BOT_PREFIX):].split()
        if not command:
            return

        cmd = command[0].lower()

        print('Command read:',cmd)

        if cmd == "speak":
            prefix = await find_prompt(channel, CONTEXT_LIMIT)
            try:
                max_len = int(command[1])
            except (IndexError, ValueError):
                max_len = LENGTH_LIMIT
            async with channel.typing():
                await generate_text(channel, prefix, R_AUTHOR, max_len=max_len, adaptive=ADAPTIVE)

    # speaking mode, auto-reply behavior
    elif BOT_MODE == 'speak' and GEN_MODE == 'auto':
        if time_last_msg is None or message.created_at - time_last_msg > datetime.timedelta(seconds=TIME_MIN_REPLY):
            memory = memory_manager_dict[guild.id][channel.id]
            time_last_msg = message.created_at
            system_prompt = f"""You are {R_AUTHOR}, a helpful storyteller who spins tales and answers questions."""
            context = memory.get_buffer()

            agent_input = await consult_agent(content)
            if agent_input:
                context += '\nHere is some background knowledge: ' + agent_input
            print(context)

            recent_conversation = memory.get_recent_conversation(user_name=message.author,bot_name=R_AUTHOR,num_rounds=3,return_separated=True)

            if openai_client:
                combined_prompt = await build_chat_prompt(cotton_tokenizer,system_prompt,content,context,recent_conversation=recent_conversation,return_dict=True)
                async with channel.typing():
                    response = await generate_openai(
                        channel=channel,
                        openai_client=openai_client,
                        prompt=combined_prompt,
                        max_new_tokens=128,
                        temperature=TEMPERATURE)
            else:
                combined_prompt = await build_chat_prompt(cotton_tokenizer,system_prompt,content,context,recent_conversation=recent_conversation,return_dict=False)
                async with channel.typing():
                    response = await generate_text(
                        channel=channel,
                        prompt=combined_prompt,
                        author=R_AUTHOR,
                        max_len=128,
                        include_prefix=False,
                        adaptive=ADAPTIVE)
                    
            memory.save_context(content,response,user_name=message.author,bot_name=R_AUTHOR)
            memory.save(path=os.path.join('memory_data',str(guild.id),str(channel.id)))

            # TTS per-server setting
            server_tts_enabled = get_server_setting(guild.id, "tts_enabled", TTS_ENABLED)
            if server_tts_enabled:
                await play_tts(author=message.author,guild=guild,text=response)
        else:
            print("Message cooldown active.")


# -------------------------------------------------------------
# Core
# -------------------------------------------------------------
async def client_loop():
    global client, client_loop_ref
    client_loop_ref = asyncio.get_running_loop()
    await client.start(BOT_TOKEN)

def init():
    global client_loop_ref, memory_manager_dict

    print('cottonbot client starting...')

    client.run(BOT_TOKEN)
    client_loop_ref = asyncio.get_running_loop()

def main(args):
    bot_mode = args.mode
    gen_mode = args.behavior
    r_author = args.author

    set_params(bot_mode, gen_mode, r_author)
    set_model_params(
        base_model=args.base_model,
        memory_model=args.memory_model,
        tts_model=args.tts_model,
        modder_model=args.modder_model,
        modder_causal_model=args.modder_causal_model
    )
    set_feature_params(
        adaptive=args.adaptive,
        temperature=args.temperature,
        tts_enabled=args.tts_enabled,
        modder_enabled=args.modder_enabled
    )
    init()  # launches client loop, etc.

if __name__ == "__main__":
    args = parse_args()
    main(args)
