import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils

import argparse
import random
from tqdm import tqdm

from models import weights_init, Discriminator, Generator
from operation import copy_Generator_parameters, load_parameters, get_dir
from operation import ImageFolder, InfiniteSamplerWrapper
from diffaug import DiffAugment

import lpips


policy = "color,translation"
percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=True)


def crop_image_by_part(image, part):
    hw = image.shape[2] // 2
    if part == 0:
        return image[:, :, :hw, :hw]
    if part == 1:
        return image[:, :, :hw, hw:]
    if part == 2:
        return image[:, :, hw:, :hw]
    if part == 3:
        return image[:, :, hw:, hw:]


def train_discriminator(net, data, label="real"):
    """Train function of discriminator"""
    if label == "real":
        part = random.randint(0, 3)
        pred, [rec_all, rec_small, rec_part] = net(data, label, part=part)
        err = (
            F.relu(torch.rand_like(pred) * 0.2 + 0.8 - pred).mean()
            + percept(rec_all, F.interpolate(data, rec_all.shape[2])).sum()
            + percept(rec_small, F.interpolate(data, rec_small.shape[2])).sum()
            + percept(
                rec_part,
                F.interpolate(crop_image_by_part(data, part), rec_part.shape[2]),
            ).sum()
        )
        err.backward()
        return pred.mean().item(), rec_all, rec_small, rec_part
    else:
        pred = net(data, label)
        err = F.relu(torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
        err.backward()
        return pred.mean().item()


def train(args):

    data_root = args.path
    total_iterations = args.iter
    checkpoint = args.ckpt
    batch_size = args.batch_size
    im_size = args.im_size
    ndf = 64
    ngf = 64
    nz = 256
    nlr = 0.0002
    nbeta1 = 0.5
    use_cuda = True
    multi_gpu = True
    dataloader_workers = args.workers
    current_iteration = args.start_iter
    save_interval = args.save_interval
    saved_model_folder, saved_image_folder = get_dir(args)

    device = torch.device("cpu")
    if use_cuda:
        device = torch.device("cuda:0")

    transform_list = [
        transforms.Resize((int(im_size), int(im_size))),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    trans = transforms.Compose(transform_list)

    dataset = ImageFolder(root=data_root, transform=trans)

    dataloader = iter(
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=InfiniteSamplerWrapper(dataset),
            num_workers=dataloader_workers,
            pin_memory=True,
        )
    )
    """
    loader = MultiEpochsDataLoader(dataset, batch_size=batch_size,
                               shuffle=True, num_workers=dataloader_workers,
                               pin_memory=True)
    dataloader = CudaDataLoader(loader, 'cuda')
    """

    # from model_s import Generator, Discriminator
    net_generator = Generator(ngf=ngf, nz=nz, im_size=im_size)
    net_generator.apply(weights_init)

    net_discriminator = Discriminator(ndf=ndf, im_size=im_size)
    net_discriminator.apply(weights_init)

    net_generator.to(device)
    net_discriminator.to(device)

    avg_param_generator = copy_Generator_parameters(net_generator)

    fixed_noise = torch.FloatTensor(8, nz).normal_(0, 1).to(device)

    optimizer_generator = optim.AdamW(net_generator.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizer_discriminator = optim.AdamW(net_discriminator.parameters(), lr=nlr, betas=(nbeta1, 0.999))

    if checkpoint != "None":
        ckpt = torch.load(checkpoint)
        net_generator.load_state_dict(
            {k.replace("module.", ""): v for k, v in ckpt["g"].items()}
        )
        net_discriminator.load_state_dict(
            {k.replace("module.", ""): v for k, v in ckpt["d"].items()}
        )
        avg_param_generator = ckpt["g_ema"]
        optimizer_generator.load_state_dict(ckpt["opt_g"])
        optimizer_discriminator.load_state_dict(ckpt["opt_d"])
        current_iteration = int(checkpoint.split("_")[-1].split(".")[0])
        del ckpt

    if multi_gpu:
        net_generator = nn.DataParallel(net_generator.to(device))
        net_discriminator = nn.DataParallel(net_discriminator.to(device))

    for iteration in tqdm(range(current_iteration, total_iterations + 1)):
        real_image = next(dataloader)
        real_image = real_image.to(device)
        current_batch_size = real_image.size(0)
        noise = torch.Tensor(current_batch_size, nz).normal_(0, 1).to(device)

        fake_images = net_generator(noise)

        real_image = DiffAugment(real_image, policy=policy)
        fake_images = [DiffAugment(fake, policy=policy) for fake in fake_images]

        # 2. train Discriminator
        net_discriminator.zero_grad()

        err_dr, rec_img_all, rec_img_small, rec_img_part = train_discriminator(
            net_discriminator, real_image, label="real"
        )
        train_discriminator(net_discriminator, [fi.detach() for fi in fake_images], label="fake")
        optimizer_discriminator.step()

        # 3. train Generator
        net_generator.zero_grad()
        pred_g = net_discriminator(fake_images, "fake")
        err_g = -pred_g.mean()

        err_g.backward()
        optimizer_generator.step()

        for p, avg_p in zip(net_generator.parameters(), avg_param_generator):
            avg_p.mul_(0.999).add_(0.001 * p.data)

        if iteration % 100 == 0:
            print("GAN: loss d: %.5f    loss g: %.5f" % (err_dr, -err_g.item()))

        if iteration % (save_interval * 10) == 0:
            backup_para = copy_Generator_parameters(net_generator)
            load_parameters(net_generator, avg_param_generator)
            with torch.no_grad():
                vutils.save_image(
                    net_generator(fixed_noise)[0].add(1).mul(0.5),
                    saved_image_folder + "/%d.jpg" % iteration,
                    nrow=4,
                )
                vutils.save_image(
                    torch.cat(
                        [
                            F.interpolate(real_image, 128),
                            rec_img_all,
                            rec_img_small,
                            rec_img_part,
                        ]
                    )
                    .add(1)
                    .mul(0.5),
                    saved_image_folder + "/rec_%d.jpg" % iteration,
                )
            load_parameters(net_generator, backup_para)

        if iteration % (save_interval * 50) == 0 or iteration == total_iterations:
            backup_para = copy_Generator_parameters(net_generator)
            load_parameters(net_generator, avg_param_generator)
            torch.save(
                {"g": net_generator.state_dict(), "d": net_discriminator.state_dict()},
                saved_model_folder + "/%d.pth" % iteration,
            )
            load_parameters(net_generator, backup_para)
            torch.save(
                {
                    "g": net_generator.state_dict(),
                    "d": net_discriminator.state_dict(),
                    "g_ema": avg_param_generator,
                    "opt_g": optimizer_generator.state_dict(),
                    "opt_d": optimizer_discriminator.state_dict(),
                },
                saved_model_folder + "/all_%d.pth" % iteration,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="region gan")

    parser.add_argument(
        "--path",
        type=str,
        default="../lmdbs/art_landscape_1k",
        help="path of resource dataset, should be a folder that has one or many sub image folders inside",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Output path for the train results",
    )
    parser.add_argument("--cuda", type=int, default=0, help="index of gpu to use")
    parser.add_argument("--name", type=str, default="test1", help="experiment name")
    parser.add_argument("--iter", type=int, default=50000, help="number of iterations")
    parser.add_argument(
        "--start_iter", type=int, default=0, help="the iteration to start training"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="mini batch number of images"
    )
    parser.add_argument("--im_size", type=int, default=256, help="image resolution")
    parser.add_argument(
        "--ckpt", type=str, default="None", help="checkpoint weight path if have one"
    )
    parser.add_argument(
        "--workers", type=int, default=2, help="number of workers for dataloader"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100,
        help="number of iterations to save model",
    )

    args = parser.parse_args()
    print(args)

    train(args)
