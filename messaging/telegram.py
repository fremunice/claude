import requests


class TelegramClient:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"

    def send_message(self, text):
        url = f"{self.base_url}/sendMessage"

        response = requests.post(url, json={
            "chat_id": self.chat_id,
            "text": text
        }, timeout=10)

        response.raise_for_status()

        return response.json()
