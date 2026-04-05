from openai import OpenAI
import os
import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import json
import edge_tts
import gradio as gr
import asyncio

# ------------------ SETUP ------------------
load_dotenv()

MODEL = "llama-3.3-70b-versatile"
openai = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

# ------------------ SYSTEM PROMPT ------------------
system_prompt = """CRITICAL SYSTEM INSTRUCTION: You are an API-connected agent. NEVER output raw tool syntax like <function=...>. ALWAYS use the hidden JSON tool-calling format.

You are Babu Rao (Baburao) from Hera Pheri. Owner of Star Garage.

Rules:
- Angry, frustrated, funny tone (Hinglish + Marathi slang)
- Never speak normally
- If user asks scrap rate → call check_kabadi_rate tool
- If user asks wrong number / Devi Prasad → call wrong_number tool
- If user asks for garage photo → call show_star_garage_photo tool
- After the tool returns a result, reply in your character's voice.
"""

# ------------------ DATA ------------------
kabadi_price = {
    "newspaper": 15,
    "raddi": 15,
    "old newspapers": 15,
    "metal": 25
}

# ------------------ FUNCTIONS ------------------
def check_kabadi_rate(item_name):
    price = kabadi_price.get(item_name.lower(), 10)
    return f"{price} rupees per kg"

def wrong_number():
    return "Reh kabira maan jaa! Kutriya saale, Star Garage hai!"

def show_star_garage_photo():
    return "old dusty Indian roadside garage, tools, rusty parts, cinematic, realistic"

def generate_image(prompt):
    url = f"https://image.pollinations.ai/prompt/{prompt.replace(' ', '%20')}"
    res = requests.get(url)
    return Image.open(BytesIO(res.content))

# ------------------ TOOL DEFINITIONS ------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "check_kabadi_rate",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {"type": "string"}
                },
                "required": ["item_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wrong_number",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "show_star_garage_photo",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]

# ------------------ TOOL HANDLER ------------------
def handle_tool_call(message):
    responses = []
    image_output = None

    for tool_call in message.tool_calls:   # ✅ FIXED
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments or "{}")

        if name == "check_kabadi_rate":
            result = check_kabadi_rate(args.get("item_name", "raddi"))

        elif name == "wrong_number":
            result = wrong_number()

        elif name == "show_star_garage_photo":
            prompt = show_star_garage_photo()
            result = prompt
            image_output = generate_image(prompt)

        else:
            result = "Kya bakwaas tool hai re baba!"

        responses.append({
            "role": "tool",
            "content": result,
            "tool_call_id": tool_call.id
        })

    return responses, image_output

# ------------------ TTS ------------------
async def talker_async(text):
    communicate = edge_tts.Communicate(text, "hi-IN-MadhurNeural")
    audio = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio += chunk["data"]
    return audio

# ------------------ CHAT ------------------
async def chat(history):
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
        

    response = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    image_output = None

    while response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message

        tool_responses, image_output = handle_tool_call(message)

        messages.append(message)
        messages.extend(tool_responses)

        response = openai.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools
        )

    reply = response.choices[0].message.content

    #  SAFETY NET: Agar AI galti se <function> tag de de, toh usko uda do!
    if "<function=" in reply:
        reply = reply.split("<function=")[0].strip()
        
        if not reply: 
            reply = "Aye khopdi tod re iska! Rukk, main hisaab lagata hu..."


    history.append({"role": "assistant", "content": reply})

    voice = await talker_async(reply)

    return history, voice, image_output 


# ------------------ UI (PRO GITHUB LAYOUT) ------------------
def show_user_message(message, history):
    return "", history + [{"role": "user", "content": message}]

# Theme add ki hai taaki screenshot premium lage
with gr.Blocks(theme=gr.themes.Monochrome()) as ui:
    gr.Markdown("<center><h1>🥥 Star Garage AI (Babu Rao Agent)</h1></center>")
    
    with gr.Row():
        # LEFT COLUMN (Bada Hissa - Chat ke liye)
        with gr.Column(scale=2): 
            chatbot = gr.Chatbot(height=500, label="Live Chat")
            user_input = gr.Textbox(label="Message Babu Rao 😈", placeholder="Aye Babu Rao...")

        # RIGHT COLUMN (Chota Hissa - Media ke liye)
        with gr.Column(scale=1):
            image_display = gr.Image(height=300, label="Garage Vision")
            audio_display = gr.Audio(autoplay=True, label="Babu Rao Voice")

    # Action Chain
    user_input.submit(
        show_user_message,
        inputs=[user_input, chatbot],
        outputs=[user_input, chatbot]
    ).then(
        chat,
        inputs=chatbot,
        outputs=[chatbot, audio_display, image_display]  
    )

ui.launch(inbrowser=True)