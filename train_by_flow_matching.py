import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from torch.utils.data import DataLoader

from tqdm import tqdm
import models
from omegaconf import OmegaConf as of

from common.registry import registry

def load_model(cfg):
    print(registry.class_name_dict)
    model_cls_name = registry.get_model_class(cfg.arch)
    return model_cls_name.from_config(cfg.config)


def main(args):
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"

    model_args, train_args, data_args = args.model, args.train, args.data
    model = load_model(model_args).to(device)

    cifar10 = CIFAR10(
        root="./data",
        train=True,
        transform=T.Compose([T.ToTensor()]),
        download=True
    )
    loader = DataLoader(
        dataset=cifar10,
        batch_size=data_args.batch_size,
        num_workers=data_args.num_workers,
        shuffle=True
    )

    optimizer = optim.Adam(
        params=model.parameters(),
        lr=train_args.learning_rate
    )

    global_step = 0
    all_steps = len(loader) * train_args.max_epoches
    print(f"Train {all_steps} steps!")

    # progress_bar = tqdm(
    #     range(0, args.max_train_steps),
    #     initial=global_step,
    #     desc="Steps",
    #     # Only show the progress bar once on each machine.
    #     # disable=not accelerator.is_local_main_process,
    # )

    for epoch in tqdm(range(1, train_args.max_epoches+1)):
        for step, batch in enumerate(loader):
            img, _ = batch
            img = img.to(device)

            bsz = img.shape[0]
            noise = torch.randn_like(img)
            u = torch.rand((bsz), device=device)

            timesteps = (u * 1000).long()
            u = u[:, None, None, None]

            x_t = (1.0 - u) * img + u * noise

            pred = model(x_t, timesteps)

            loss = ((pred - (noise - x_t))**2).mean()
            loss.backward()

            optimizer.step()

            global_step += 1

            if global_step % train_args.validation_step == 0:
                print(f"{epoch}: {loss.item()}")

            if global_step % train_args.checkpoint_step == 0:
                torch.save({
                    "model": model.state_dict()
                }, f"checkpoint-{global_step}.pth.tar.gz")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-file", type=str, default="./configs/flow_matching.yaml")

    args = parser.parse_args()

    args = of.load(args.cfg_file)

    print(args)

    main(args)