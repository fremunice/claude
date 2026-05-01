import os

from dotenv import load_dotenv

from messaging.telegram import TelegramClient

load_dotenv(override=True)

telegram = TelegramClient(
    token=os.getenv("TELEGRAM_BOT_TOKEN"),
    chat_id=os.getenv("ALLOWED_TELEGRAM_USER_ID"),
)

res = telegram.send_message("Test Telegram berhasil connect")
print(res)
