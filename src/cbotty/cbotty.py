import time
from nicegui import ui

from workflow.agents import SharedResources, invoke
from utils.Configuration import Configuration
from dotenv import load_dotenv

load_dotenv()

session_id = str(time.time())
config = Configuration()
context = SharedResources.create(config=config)

chat_area = ui.column().classes('w-full h-[80vh] overflow-auto p-4 border rounded-lg gap-2')

async def generate_stream(prompt: str):
    """Simulate streaming response"""

    async for chunk in invoke(question=prompt, context=context, session_id=session_id):
        yield chunk

def classify_chunk(chunk):
    if not chunk:
        return "Sorry, I have encountered an error.<br>"
    if isinstance(chunk, str):
        return chunk
    key = list(chunk.keys())[0]
    if not chunk[key]:
        return "Sorry, I have encountered an error. Retrying...<br>"
    match key:
        case 'get_sentiment':
            return f"<span style='color: grey;'>The user has shown a {chunk[key]['sentiment']} sentiment.</span><br>"
        case 'department_router':
            return f"<span style='color: grey;'>Sending request to department: {chunk[key]['department']}</span><br>"
        case 'handle_request' | 'customer_agent':
            return f"{chunk[key]['response']}<br>"
        case _:
            return str(chunk)

async def send_message(*_):
    msg = user_input.value.strip()
    if not msg:
        return
    user_input.value = ''

    # --- User message (right-aligned) ---
    with chat_area:
        with ui.row().classes('justify-end items-start gap-2'):
            ui.label(msg).classes('bg-blue-500 text-white p-2 rounded-lg max-w-xs break-words')
            ui.label('🧑')  # avatar

    # --- Bot message (left-aligned) ---
    with chat_area:
        with ui.row().classes('justify-start items-start gap-2') as bot_row:
            bot_markdown = ui.markdown('')
            bot_markdown.classes('bg-gray-200 text-black p-2 rounded-lg break-words')
            # bot_markdown.classes('bg-gray-200 text-black p-2 rounded-lg max-w-xs break-words')
            ui.label('🤖')  # avatar

    # Stream the bot response
    text_so_far = ''
    async for chunk in generate_stream(msg):
        text_so_far += classify_chunk(chunk)
        bot_markdown.content = text_so_far

# --- Input row ---
with ui.row().classes('w-full items-center gap-2'):
    user_input = ui.input(placeholder='Type your message...').classes('flex-grow')
    user_input.on('keydown.enter', send_message)
    ui.button('Send', on_click=send_message, icon='send').props('color=primary rounded outline')

ui.run(title='RAG Chat Demo', dark=True, host='0.0.0.0', port=8080)
