from nicegui import ui
import asyncio

chat_area = ui.column().classes('w-full h-[80vh] overflow-auto p-4 border rounded-lg gap-2')

async def generate_stream(prompt: str):
    """Simulate streaming response"""
    response = f"That's an interesting thought about '{prompt}'. Let me elaborate..."
    for token in response.split():
        yield token + ' '
        await asyncio.sleep(0.1)

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
            bot_label = ui.label('')  # text will stream here
            bot_label.classes('bg-gray-200 text-black p-2 rounded-lg max-w-xs break-words')
            ui.label('🤖')  # avatar

    # Stream the bot response
    text_so_far = ''
    async for chunk in generate_stream(msg):
        text_so_far += chunk
        bot_label.set_text(text_so_far)

# --- Input row ---
with ui.row().classes('w-full items-center gap-2'):
    user_input = ui.input(placeholder='Type your message...').classes('flex-grow')
    user_input.on('keydown.enter', send_message)
    ui.button('Send', on_click=send_message, icon='send').props('color=primary rounded outline')

ui.run(title='RAG Chat Demo', dark=True, host='0.0.0.0', port=8080)
