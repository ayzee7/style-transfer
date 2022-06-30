from PIL import Image
import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
from adain_model import Model, denormalize


def transfer(grade, content_resize=False, style_resize=False):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

    content = './data/images/neural-style/content.jpg'
    style = './data/images/neural-style/style.jpg'
    out = './data/images/neural-style/result.jpg'
    model_state = './data/model_state.pth'
    alpha = grade      # defines style transfer grade


    print('transforming photos')
    c = Image.open(content)
    os.remove(content)
    c_transform = transforms.Compose([transforms.Resize((int(0.7 * c.size[1]), int(0.7 * c.size[0]))),
                                      transforms.ToTensor(),
                                      normalize])


    s = Image.open(style)
    os.remove(style)
    s_transform = transforms.Compose([transforms.Resize((int(0.7 * s.size[1]), int(0.7 * s.size[0]))),
                                      transforms.ToTensor(),
                                      normalize])


    # set device on GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('setting model')
    model = Model()
    if model_state is not None:
        model.load_state_dict(torch.load(model_state, map_location=lambda storage, loc: storage))
    model = model.to(device)

    c_tensor = c_transform(c).unsqueeze(0).to(device)
    s_tensor = s_transform(s).unsqueeze(0).to(device)

    print('generating photo')
    with torch.no_grad():
        output = model.generate(c_tensor, s_tensor, alpha)

    output = denormalize(output, device)

    print('saving image')
    save_image(output, out)

