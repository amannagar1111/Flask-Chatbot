from flask import Flask, render_template, request

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form["msg"]
    return get_Chat_response(user_input)

def get_Chat_response(text):
    greetings = ["hello", "hi", "hey", "namaste"]
    medical_queries = {
        "fever": "I'm sorry to hear that you're experiencing a fever. It's essential to rest, stay hydrated, and take a fever reducer like acetaminophen. If it persists or worsens, please consult a healthcare professional.",
        "headache": "Headaches can be caused by various factors. Try resting in a quiet, dark room and consider taking over-the-counter pain relievers. If headaches persist or worsen, consult a doctor.",
        # Add more medical queries and responses here
    }

    contact_reply = "For any inquiries, you can contact us at baymaxhealthcare123@gmail.com."

    if any(greeting in text.lower() for greeting in greetings):
        return "Hello! Welcome to Baymax Healthcare <3."

    for query, response in medical_queries.items():
        if query in text.lower():
            return response

    if "contact" in text.lower() or "details" in text.lower():
        return contact_reply

    bot_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

if __name__ == '__main__':
    app.run()
