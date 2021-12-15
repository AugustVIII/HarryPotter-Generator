from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import message, user

import textwrap
import numpy as np
import torch
from transformers import GPT2Tokenizer

import logging

API_TOKEN = '2131697027:AAEWB_aT-SJ5c17itraDFSrfdZ-1yfdZUuo'

STICKER_1 = 'CAACAgQAAxkBAANUYZJIzXAkGJLIC1r04SKRU1t0FVwAAooJAAJIhgABUuKIVYrc94mfIgQ'
sticker_2 = 'CAACAgQAAxkBAAEDU9BhmPvLKfE48348FCZavlXFfG7mPAACEQkAAue58FHvm_UQ0HB2LSIE'
sticker_3 = 'CAACAgQAAxkBAAEDU9JhmPvOOlk3LfxtiTqx5mx5cINMHgACmAsAAr1j8VGvIa3QVPVb_yIE'
sticker_4 = 'CAACAgQAAxkBAAEDU9RhmPvRQ805JTL5cawkz8GrwVErhwACkAkAAlY48FGA7F_CGXNo6iIE'
sticker_5 = 'CAACAgQAAxkBAAEDU9ZhmPwiqi9qTixGMZE9er3v3CflRwAC9wwAAkEo8VGLI--t-4-xtiIE'
sticker_6 = 'CAACAgQAAxkBAAEDU9hhmPwmKPL3FcRjpUs70sBwMMoe2AACWgoAAnEt8VEr2SLb0UUakiIE'
sticker_7 = 'CAACAgQAAxkBAAEDU9phmPxBtOzPJ-S9xpkf_CfV3zZzkAACPgwAAveI8VGBSDdYdUK0fSIE'
sticker_8 = 'CAACAgQAAxkBAAEDU9xhmPxINikLCFEkJQVFW-ColQFgawACrAoAApeR8FEolJj7CfpxtSIE'
sticker_9 = 'CAACAgQAAxkBAAEDU91hmPxJHoWmcFxcjxjLEG504aj-dQACdgoAAq4j6VGhUbXAqn6wtSIE'
sticker_10 = 'CAACAgQAAxkBAAEDU-BhmPybmthfgGljg-6iyVLlM73FcAACqwkAAk8B-VETPFWYNc_rfSIE'
sticker_11 = 'CAACAgQAAxkBAAEDU-FhmPyc2OnWlNiPN1y79sfH3qBSuAAC4AsAAqXtAVKc0gZY8nsslSIE'
sticker_12 = 'CAACAgQAAxkBAAEDU-RhmPyfeOjvYDEN8WPXBiCuoHxSWwACAwoAAqfG-FGCB9SPFbp8ZSIE'
sticker_13 = 'CAACAgQAAxkBAAEDU-ZhmPylcmh_i_NtSongml9LTlqOKAAC5AgAAnxNCFLCrC4jBfdz6iIE'

#user
# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)


#Load the model to use it
model = torch.load('/home/mark/Documents/GitHub/00_Projects/Project_5.NLP HarryPotterAI/model_ru.pt', map_location=torch.device('cpu'))
tokenizer = GPT2Tokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')


def generate_v2(prompt, len=150, t=2.):
    generated = tokenizer.encode(prompt, return_tensors='pt')
    out = model.generate(
        input_ids=generated,
        max_length=len,
        num_beams=7,
        do_sample=True,
        temperature=t,
        top_k=50,
        top_p=0.7,
        no_repeat_ngram_size=2,
        num_return_sequences=1,
        ).cpu().numpy()

    sequence = textwrap.fill(tokenizer.decode(*out), len)
    return sequence

@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.reply("Привет,\nЯ гененирую тексты в стиле Гарри Поттера!\n \
        Отправь мне слово/фразу/предложение и я выдам тебе сгенерированный текст. \
        Список команд \
         \n/start - перезапуск\n \
         \n/info - расскажу о себе\n \
         \n/help - список доступных команд\n")


@dp.message_handler(commands=['help'])
async def send_welcome(message: types.Message):
    await message.reply("start - перезапуск\n \
                        \n/info - расскажу о себе\n \
                        \n/help - список доступных команд\n")

@dp.message_handler(commands=['info'])
async def send_welcome(message: types.Message):
    await message.reply("В основу моего обучения легли: \
                        \n 1) Предобученная модель GPT-2 от Сбербанка - sberbank-ai/rugpt3small_based_on_gpt2 \n \
                        \n 2) Серия романов о Гарри Поттере (перевод Росмэн) \n ") 

@dp.message_handler()
async def scale(message: types.Message):

    #Send sticker
    user_id = message.from_user.id
    await bot.send_sticker(user_id, STICKER_1)

    #Generate text
    mes = textwrap.fill(generate_v2(message.text, len=150, t=2.), 120)
    await message.reply(mes)

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)