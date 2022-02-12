import telebot
import os
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


showup = types.ReplyKeyboardMarkup(one_time_keyboard=True)
showup.add(types.KeyboardButton('/transfer'))
# stop = types.ReplyKeyboardMarkup(one_time_keyboard=True)
# stop.add(types.KeyboardButton('/cancel'))

@bot.message_handler(commands=['start'])
def hello(message):
    bot.send_message(message.chat.id, 'Hello! Welcome to Style Transfer Bot. '
                                      'Type in /transfer to start the bot.', reply_markup=showup)


@bot.message_handler(commands=['transfer'])
def style_in(message):
    msg = bot.reply_to(message, 'Send me a photo to copy the style from')
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
        bot.reply_to(message, 'You have to send a photo, not a text. Try again with /transfer', reply_markup=showup)
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
        bot.reply_to(message, 'You have to send a photo, not a text. Try again with /transfer', reply_markup=showup)
        return None
    try:
        transfer()
        src = './data/images/neural-style/result.jpg'
        res = open(src, 'rb')
        bot.send_photo(message.chat.id, res)
        res.close()
        os.remove(src)
        bot.send_message(message.chat.id, 'Wanna try one more time?', reply_markup=showup)
    except AssertionError:
        bot.send_message(message.chat.id, 'Sorry, the provided photos cannot be used together due to their size. Please try the other ones')
    except IOError:
        return None


"""@bot.message_handler(commands=['cancel'])
def cancel_message(message):
    bot.send_message(message.chat.id, 'Stopping the bot...', reply_markup=showup)
    src = './data/images/neural-style/.stop'
    with open(src, 'w') as stop:
        stop.write('1')"""


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


