import os
import base64
import json
from io import BytesIO
import httpx
import requests
from dotenv import load_dotenv
from g4f.client import Client
from duckduckgo_search import DDGS

# ─────────────────────────────────────────────
# Load environment variables
# ─────────────────────────────────────────────
load_dotenv()
OWNER_ID = os.getenv("OWNER_ID")  # string; compare as str

# ─────────────────────────────────────────────
# Global Settings
# ─────────────────────────────────────────────
current_option = 1                     # Default model option
owner_temp_option: int | None = None   # Temporary override for owner only
deepinfra_requests = 0                 # Request counter
g4f_requests = 0                       # Request counter

# Reuse existing async client (for codeltix image endpoint)
cl = httpx.AsyncClient(base_url='https://api.codeltix.com',
                       follow_redirects=True, timeout=20)

# ─────────────────────────────────────────────
# Model Option 1 – DeepInfra (openai/gpt-oss-120b)
# ─────────────────────────────────────────────
async def deepinfra_response(history: list[dict[str, str]]) -> str:
    """Get AI response from DeepInfra API (Option 1)."""
    global deepinfra_requests
    try:
        url = "https://api.deepinfra.com/v1/openai/chat/completions"
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "origin": "https://deepinfra.com",
            "referer": "https://deepinfra.com/",
            "user-agent": "Mozilla/5.0",
            "x-deepinfra-source": "web-pag",
            # "Authorization": f"Bearer {os.getenv('DEEPINFRA_API_KEY')}",  # optional
        }

        payload = {
            "model": "openai/gpt-oss-120b",
            "messages": history,
            "stream": False
        }

        response = requests.post(url, headers=headers, data=json.dumps(payload))
        data = response.json()
        deepinfra_requests += 1
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"DeepInfra error: {e}")
        return "I'm sorry, I couldn't get a response from DeepInfra right now."


# ─────────────────────────────────────────────
# Model Option 2 – g4f + DuckDuckGo fallback
# ─────────────────────────────────────────────
async def g4f_response(history: list[dict[str, str]]) -> str:
    """Get AI response from g4f with DuckDuckGo fallback (Option 2)."""
    global g4f_requests
    try:
        client = Client()
        prompt = history[-1]["content"] if history else "Hello"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=history
        ).choices[0].message.content
        g4f_requests += 1

        # Fallback to DuckDuckGo if too short or unsure
        if "I don't know" in response or len(response) < 20:
            search_result = DDGS().text(prompt)
            web_text = "\n".join([r["body"] for r in search_result[:3]])
            combined_prompt = f"{prompt}\n\nUse this web info to answer:\n{web_text}"
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": combined_prompt}]
            ).choices[0].message.content

        return response
    except Exception as e:
        print(f"g4f error: {e}")
        return f"Error: {e}"


# ─────────────────────────────────────────────
# Main Unified Function – Uses current option
# ─────────────────────────────────────────────
async def get_ai_response(history: list[dict[str, str]], user_id: str | int = None) -> str:
    """
    Get the AI response for the given history.
    Automatically uses current option or owner's temporary setting.
    """
    global current_option, owner_temp_option
    option_to_use = current_option

    # Owner temporary option (if active)
    if user_id and str(user_id) == str(OWNER_ID) and owner_temp_option:
        option_to_use = owner_temp_option

    if option_to_use == 2:
        return await g4f_response(history)
    else:
        return await deepinfra_response(history)


# ─────────────────────────────────────────────
# Option Command Handling
# ─────────────────────────────────────────────
async def handle_option_command(user_id: str | int, command: str) -> str:
    """
    Handle /tryoption or /fixoption commands.
    Only owner can change or test model options.
    """
    global current_option, owner_temp_option

    if str(user_id) != str(OWNER_ID):
        return "⛔ Only the owner can use this command."

    parts = command.strip().split()
    if len(parts) != 2 or parts[0] not in ["/tryoption", "/fixoption"]:
        return "Usage: /tryoption <1|2> or /fixoption <1|2>"

    cmd, value = parts[0], parts[1]
    if value not in ["1", "2"]:
        return "❌ Invalid option. Choose 1 or 2."

    value = int(value)

    if cmd == "/tryoption":
        owner_temp_option = value
        return f"✅ Temporary model switched to Option {value} (for owner only)."
    else:
        current_option = value
        owner_temp_option = None
        return f"✅ Global model permanently changed to Option {value}."


# ─────────────────────────────────────────────
# Model Status Command
# ─────────────────────────────────────────────
async def handle_status_command(user_id: str | int) -> str:
    """
    Show model usage stats and current active model.
    """
    global current_option, owner_temp_option, deepinfra_requests, g4f_requests

    option_display = f"Option 1 → DeepInfra\nTotal Requests: {deepinfra_requests}\n\n" \
                     f"Option 2 → g4f + DuckDuckGo\nTotal Requests: {g4f_requests}\n\n"

    if str(user_id) == str(OWNER_ID) and owner_temp_option:
        current = f"Current Model: Option {owner_temp_option} (temporary for owner)"
    else:
        current = f"Current Model: Option {current_option}"

    return f"📊 Model Status\n\n{option_display}{current}"


# ─────────────────────────────────────────────
# Image Generation (unchanged)
# ─────────────────────────────────────────────
async def create_image(encoded_prompt: str) -> BytesIO | None:
    """
    Get an image representation for the given prompt.
    (Unchanged from original.)
    """
    try:
        response = await cl.get(f"/ai/image/?prompt={encoded_prompt}")
        response.raise_for_status()
        base64_image = response.json()["image"]
        image_data = base64.b64decode(base64_image)
        image_file = BytesIO(image_data)
        return image_file
    except httpx.HTTPStatusError as exc:
        print(f'HTTP error occurred: {exc}')
    except httpx.RequestError as exc:
        print(f'Request error occurred: {exc}')
    except Exception as exc:
        print(f'An unexpected error occurred: {exc}')
    return None
