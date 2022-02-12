import warnings
warnings.simplefilter("ignore", UserWarning)
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from adain_dataset import PreprocessDataset
from adain_model import Model, denormalize

def main():
    batch_size = 8
    epoch = 20
    lr = 1e-6
    train_content_dir = './data/train/content'
    train_style_dir = './data/train/style'
    test_content_dir = './data/test/content'
    test_style_dir = './data/test/style'
    snap_int = 1000
    save_dir = './data/save'

    # set device on GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f'# batch-size: {batch_size}')
    print(f'# epoch: {epoch}')
    print('')

    # prepare dataset and dataLoader
    train_dataset = PreprocessDataset(train_content_dir, train_style_dir)
    test_dataset = PreprocessDataset(test_content_dir, test_style_dir)
    iters = len(train_dataset)
    print(f'Length of train image pairs: {iters}')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_iter = iter(test_loader)

    # set model and optimizer
    model = Model().to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    # start training
    loss_list = []
    for e in range(1, epoch + 1):
        print(f'Start {e} epoch')
        for i, (content, style) in tqdm(enumerate(train_loader, 1)):
            content = content.to(device)
            style = style.to(device)
            loss = model(content, style)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'[{e}/total {epoch} epoch],[{i} /'
                  f'total {round(iters/batch_size)} iteration]: {loss.item()}')

            if i % snap_int == 0:
                content, style = next(test_iter)
                content = content.to(device)
                style = style.to(device)
                with torch.no_grad():
                    out = model.generate(content, style)
                content = denormalize(content, device)
                style = denormalize(style, device)
                out = denormalize(out, device)
                res = torch.cat([content, style, out], dim=0)
                res = res.to('cpu')
                save_image(res, f'{save_dir}/{e}_epoch_{i}_iteration.png', nrow=batch_size)
        torch.save(model.state_dict(), f'{save_dir}/{e}_epoch.pth')
    plt.plot(range(len(loss_list)), loss_list)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('train loss')
    plt.savefig(f'{save_dir}/train_loss.png')
    with open(f'{save_dir}/loss_log.txt', 'w') as f:
        for l in loss_list:
            f.write(f'{l}\n')
    print(f'Loss saved in {save_dir}')


if __name__ == '__main__':
    main()