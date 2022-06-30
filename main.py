import telebot
import os
import datetime as dt
from telebot import types
import logging
from flask import Flask, request
from transfer_adain import transfer
import config

TOKEN = config.TOKEN
WEBHOOK_HOST = config.WEBHOOK_HOST

WEBHOOK_PORT = os.environ.get('PORT', 5000)
WEBHOOK_LISTEN = '0.0.0.0'


WEBHOOK_URL_BASE = "https://{}".format(WEBHOOK_HOST)
WEBHOOK_URL_PATH = "/{}".format(TOKEN)

logger = telebot.logger
telebot.logger.setLevel(logging.INFO)

bot = telebot.TeleBot(TOKEN)

server = Flask(__name__)

class Grade:
    grade = 1         # defines morph grade

markup = types.InlineKeyboardMarkup()
button_transfer = types.InlineKeyboardButton("Transfer", callback_data="tr")
button_morph = types.InlineKeyboardButton("Change morph grade", callback_data="mg")
markup.row(button_transfer)
markup.row(button_morph)
remove = types.ReplyKeyboardRemove(selective=False)

@bot.message_handler(commands=['start'])
def hello(message):
    bot.send_message(message.chat.id, 'Hello! Welcome to Style Transfer Bot. '
                                      'Type in /transfer to start the bot.', reply_markup=markup)

@bot.message_handler()
def style_in(message):
     if message.text == '/transfer':
        msg = bot.reply_to(message, 'Send me a photo to copy the style from', reply_markup=remove)
        bot.register_next_step_handler(msg, handle_style_photo)


def handle_style_photo(message):
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        src = './data/images/neural-style/style.jpg'
        with open(src, 'wb') as style:
            style.write(downloaded_file)
        msg = bot.reply_to(message, 'Send me a photo to copy the content from')
        bot.register_next_step_handler(msg, handle_content_photo)
    except TypeError:
        bot.reply_to(message, 'You have to send a photo. Try again with /transfer', reply_markup=markup)
        return None


def handle_content_photo(message):
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        src = './data/images/neural-style/content.jpg'
        with open(src, 'wb') as content:
            content.write(downloaded_file)
        bot.reply_to(message, 'Processing...') #, reply_markup=stop)
    except TypeError:
        bot.reply_to(message, 'You have to send a photo. Try again with /transfer', reply_markup=markup)
        return None
    try:
        transfer(Grade.grade)
        src = './data/images/neural-style/result.jpg'
        res = open(src, 'rb')
        bot.send_photo(message.chat.id, res)
        res.close()
        os.remove(src)
        bot.send_message(message.chat.id, 'Wanna try one more time?', reply_markup=markup)
    except SystemError:
        bot.reply_to(message, 'Sorry, picture size is too big for bot to handle. Please, try another one', reply_markup=markup)
    except IOError:
        return None

@bot.message_handler()
def change_morph(message):
    if message.text == '/morph':
        msg = bot.reply_to(message, 'Type in the number from 0 to 1, which defines style transfer grade', reply_markup=remove)
        bot.register_next_step_handler(msg, morph_reply)

def morph_reply(message):
    try:
        Grade.grade = float(message.text)
        bot.reply_to(message, f"Style transfer grade was successfully changed to {Grade.grade}", reply_markup=markup)
    except Exception:
        bot.reply_to(message, 'Input is incorrect. Try again', reply_markup=markup)


def send_transfer(message):
    new_message = telebot.types.Message(message_id=message.message_id,
                                        chat=message.chat,
                                        content_type=["text"],
                                        date=dt.datetime.today().timestamp(),
                                        from_user=message.chat,
                                        options={},
                                        json_string="")
    new_message.text = '/transfer'
    style_in(new_message)

def send_morph(message):
    new_message = telebot.types.Message(message_id=message.message_id,
                                        chat=message.chat,
                                        content_type=["text"],
                                        date=dt.datetime.today().timestamp(),
                                        from_user=message.chat,
                                        options={},
                                        json_string="")
    new_message.text = '/morph'
    change_morph(new_message)

@bot.callback_query_handler(lambda call: True)
def style_in_call(call):
    if call.data == "tr":
        send_transfer(call.message)
    elif call.data == "mg":
        send_morph(call.message)


@server.route('/' + TOKEN, methods=['POST'])
def get_message():
    json_string = request.get_data().decode('utf-8')
    update = telebot.types.Update.de_json(json_string)
    bot.process_new_updates([update])
    return "!", 200


@server.route("/")
def webhook():
    bot.remove_webhook()
    bot.set_webhook(url=WEBHOOK_URL_BASE + WEBHOOK_URL_PATH)
    return "!", 200


if __name__ == "__main__":
    server.run(host=WEBHOOK_LISTEN, port=WEBHOOK_PORT)


