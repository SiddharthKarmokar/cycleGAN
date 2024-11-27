import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import config
from src import utils
from src.utils.Generator import Generator
from src.utils.Discriminator import Discriminator
from src.utils.Dataset import ImageSet
import torch.optim as optim
from tqdm import tqdm

dataset = ImageSet(config.PHOTO_PATH, config.MONET_PATH, config.transform)
dataloader = DataLoader(dataset, config.BATCH_SIZE, shuffle=True)

disc_photo = Discriminator().to(config.DEVICE)
disc_monet = Discriminator().to(config.DEVICE)


gen_photo = Generator().to(config.DEVICE)
gen_monet = Generator().to(config.DEVICE)


disc_optimizer = optim.Adam(
    list(disc_photo.parameters()) + list(disc_monet.parameters()),
    lr = config.LEARNING_RATE,
    betas=(0.5, 0.999)
)

gen_optimizer = optim.Adam(
    list(gen_photo.parameters()) + list(gen_monet.parameters()),
    lr = config.LEARNING_RATE,
    betas=(0.5, 0.999)
)

dis_scaler = torch.amp.GradScaler('cuda')
gen_scaler = torch.amp.GradScaler('cuda')

MSE = nn.MSELoss()
L1 = nn.L1Loss()

for epoch in range(config.EPOCHES):
    running_dis_loss = 0.0
    running_gen_loss = 0.0
    for photo, monet in tqdm(dataloader, leave=True):
        photo = photo.to(config.DEVICE)
        monet = monet.to(config.DEVICE)

        with torch.cuda.amp.autocast():
            fake_photo = gen_photo(monet)
            dis_photo_real = disc_photo(photo)
            dis_photo_fake = disc_photo(fake_photo.detach())
            dis_photo_loss = MSE(dis_photo_real, torch.ones_like(dis_photo_real)) + \
                             MSE(dis_photo_fake, torch.zeros_like(dis_photo_fake))

            fake_monet = gen_monet(photo)
            dis_monet_real = disc_monet(monet)
            dis_monet_fake = disc_monet(fake_monet.detach())
            dis_monet_loss = MSE(dis_monet_real, torch.ones_like(dis_monet_real)) + \
                             MSE(dis_monet_fake, torch.zeros_like(dis_monet_fake))

            dis_loss = (dis_photo_loss + dis_monet_loss) / 2.0
            running_dis_loss += dis_loss / len(dataloader)

        disc_optimizer.zero_grad()
        dis_scaler.scale(dis_loss).backward()
        dis_scaler.step(disc_optimizer)
        dis_scaler.update()

        with torch.cuda.amp.autocast():
            dis_photo_fake = disc_photo(fake_photo)
            dis_monet_fake = disc_monet(fake_monet)

            gen_photo_loss = MSE(dis_photo_fake, torch.ones_like(dis_photo_fake))
            gen_monet_loss = MSE(dis_monet_fake, torch.zeros_like(dis_monet_fake))

            cycled_monet = gen_monet(fake_photo)
            cycled_photo = gen_photo(fake_monet)

            cycled_loss = L1(monet, cycled_monet) + L1(photo, cycled_photo)
            gen_loss = gen_photo_loss + gen_monet_loss + cycled_loss * config.LAMBDA_CYCLE
            running_gen_loss += gen_loss / len(dataloader)

        gen_optimizer.zero_grad()
        gen_scaler.scale(gen_loss).backward()
        gen_scaler.step(gen_optimizer)
        gen_scaler.update()

    print(f"Epoch {epoch + 1}. Generator loss by epoch: {running_gen_loss} discriminator_loss by epoch: {running_dis_loss}")