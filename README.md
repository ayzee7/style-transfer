# Style Transfer Telegram Bot Project

Telegram bot with style transfer function using AdaIN model [(source code)](https://github.com/irasin/Pytorch_AdaIN)

Bot is currently deployed at Heroku and is available as <b>@styletransfbot</b> in Telegram.

More project commentary and additional information can be found [here](/report.md) <b>(only RU version)</b>

### Usage

Start the bot with the `/start` command as usual. Then follow the provided instructions.

> Note: photo generation process and answering to `/transfer` command can take a while

### Code
- `main.py` used to run the bot
- `adain_model.py` contains the NST model itself
- `transfer_adain.py` generates the output image

I also provided the train script from source code and [the slower version of NST algorithm using VGG19 pretrained model](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html) which was used at early development stages at `old_train` folder

Bot work example and model state can be found in `data`

### Updates and fixes to be made:

- [ ] Add the option to choose the style transfer degree <i>(currently is set by default as `alpha = 1`)</i>
- [ ] Add async
