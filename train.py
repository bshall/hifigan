import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from hifigan.generator import HifiganGenerator
from hifigan.discriminator import (
    HifiganDiscriminator,
    feature_loss,
    discriminator_loss,
    generator_loss,
)
from hifigan.dataset import MelDataset, LogMelSpectrogram
from hifigan.utils import load_checkpoint, save_checkpoint, plot_spectrogram


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


BATCH_SIZE = 8
SEGMENT_LENGTH = 8320
HOP_LENGTH = 160
SAMPLE_RATE = 16000
BASE_LEARNING_RATE = 2e-4
FINETUNE_LEARNING_RATE = 1e-4
BETAS = (0.8, 0.99)
LEARNING_RATE_DECAY = 0.999
WEIGHT_DECAY = 1e-5
EPOCHS = 3100
LOG_INTERVAL = 5
VALIDATION_INTERVAL = 1000
NUM_GENERATED_EXAMPLES = 10
CHECKPOINT_INTERVAL = 5000


def train_model(rank, world_size, args):
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        init_method="tcp://localhost:54321",
    )

    log_dir = args.checkpoint_dir / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)

    if rank == 0:
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(log_dir / f"{args.checkpoint_dir.stem}.log")
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%m/%d/%Y %I:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        logger.setLevel(logging.ERROR)

    writer = SummaryWriter(log_dir) if rank == 0 else None

    generator = HifiganGenerator().to(rank)
    discriminator = HifiganDiscriminator().to(rank)

    generator = DDP(generator, device_ids=[rank])
    discriminator = DDP(discriminator, device_ids=[rank])

    optimizer_generator = optim.AdamW(
        generator.parameters(),
        lr=BASE_LEARNING_RATE if not args.finetune else FINETUNE_LEARNING_RATE,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )
    optimizer_discriminator = optim.AdamW(
        discriminator.parameters(),
        lr=BASE_LEARNING_RATE if not args.finetune else FINETUNE_LEARNING_RATE,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler_generator = optim.lr_scheduler.ExponentialLR(
        optimizer_generator, gamma=LEARNING_RATE_DECAY
    )
    scheduler_discriminator = optim.lr_scheduler.ExponentialLR(
        optimizer_discriminator, gamma=LEARNING_RATE_DECAY
    )

    train_dataset = MelDataset(
        root=args.dataset_dir,
        segment_length=SEGMENT_LENGTH,
        sample_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        train=True,
        finetune=args.finetune,
    )
    train_sampler = DistributedSampler(train_dataset, drop_last=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=8,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
    )

    validation_dataset = MelDataset(
        root=args.dataset_dir,
        segment_length=SEGMENT_LENGTH,
        sample_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        train=False,
        finetune=args.finetune,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    melspectrogram = LogMelSpectrogram().to(rank)

    if args.resume is not None:
        global_step, best_loss = load_checkpoint(
            load_path=args.resume,
            generator=generator,
            discriminator=discriminator,
            optimizer_generator=optimizer_generator,
            optimizer_discriminator=optimizer_discriminator,
            scheduler_generator=scheduler_generator,
            scheduler_discriminator=scheduler_discriminator,
            rank=rank,
            logger=logger,
            finetune=args.finetune,
        )
    else:
        global_step, best_loss = 0, float("inf")

    if args.finetune:
        global_step, best_loss = 0, float("inf")

    n_epochs = EPOCHS
    start_epoch = global_step // len(train_loader) + 1

    logger.info("**" * 40)
    logger.info(f"batch size: {BATCH_SIZE}")
    logger.info(f"iterations per epoch: {len(train_loader)}")
    logger.info(f"total of epochs: {n_epochs}")
    logger.info(f"started at epoch: {start_epoch}")
    logger.info("**" * 40 + "\n")

    for epoch in range(start_epoch, n_epochs + 1):
        train_sampler.set_epoch(epoch)

        generator.train()
        discriminator.train()
        average_loss_mel = average_loss_discriminator = average_loss_generator = 0
        for i, (wavs, mels, tgts) in enumerate(train_loader, 1):
            wavs, mels, tgts = wavs.to(rank), mels.to(rank), tgts.to(rank)

            # Discriminator
            optimizer_discriminator.zero_grad()

            wavs_ = generator(mels.squeeze(1))
            mels_ = melspectrogram(wavs_)

            scores, _ = discriminator(wavs)
            scores_, _ = discriminator(wavs_.detach())

            loss_discriminator, _, _ = discriminator_loss(scores, scores_)

            loss_discriminator.backward()
            optimizer_discriminator.step()

            # Generator
            optimizer_generator.zero_grad()

            scores, features = discriminator(wavs)
            scores_, features_ = discriminator(wavs_)

            loss_mel = F.l1_loss(mels_, tgts)
            loss_features = feature_loss(features, features_)
            loss_generator_adversarial, _ = generator_loss(scores_)
            loss_generator = 45 * loss_mel + loss_features + loss_generator_adversarial

            loss_generator.backward()
            optimizer_generator.step()

            global_step += 1

            average_loss_mel += (loss_mel.item() - average_loss_mel) / i
            average_loss_discriminator += (
                loss_discriminator.item() - average_loss_discriminator
            ) / i
            average_loss_generator += (
                loss_generator.item() - average_loss_generator
            ) / i

            if rank == 0:
                if global_step % LOG_INTERVAL == 0:
                    writer.add_scalar(
                        "train/loss_mel",
                        loss_mel.item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/loss_generator",
                        loss_generator.item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/loss_discriminator",
                        loss_discriminator.item(),
                        global_step,
                    )

            if global_step % VALIDATION_INTERVAL == 0:
                generator.eval()

                average_validation_loss = 0
                for j, (wavs, mels, tgts) in enumerate(validation_loader, 1):
                    wavs, mels, tgts = wavs.to(rank), mels.to(rank), tgts.to(rank)

                    with torch.no_grad():
                        wavs_ = generator(mels.squeeze(1))
                        mels_ = melspectrogram(wavs_)

                        length = min(mels_.size(-1), tgts.size(-1))

                        loss_mel = F.l1_loss(mels_[..., :length], tgts[..., :length])

                    average_validation_loss += (
                        loss_mel.item() - average_validation_loss
                    ) / j

                    if rank == 0:
                        if j <= NUM_GENERATED_EXAMPLES:
                            writer.add_audio(
                                f"generated/wav_{j}",
                                wavs_.squeeze(0),
                                global_step,
                                sample_rate=16000,
                            )
                            writer.add_figure(
                                f"generated/mel_{j}",
                                plot_spectrogram(mels_.squeeze().cpu().numpy()),
                                global_step,
                            )

                generator.train()
                discriminator.train()

                if rank == 0:
                    writer.add_scalar(
                        "validation/mel_loss", average_validation_loss, global_step
                    )
                    logger.info(
                        f"valid -- epoch: {epoch}, mel loss: {average_validation_loss:.4f}"
                    )

                new_best = best_loss > average_validation_loss
                if new_best or global_step % CHECKPOINT_INTERVAL == 0:
                    if new_best:
                        logger.info("-------- new best model found!")
                        best_loss = average_validation_loss

                    if rank == 0:
                        save_checkpoint(
                            checkpoint_dir=args.checkpoint_dir,
                            generator=generator,
                            discriminator=discriminator,
                            optimizer_generator=optimizer_generator,
                            optimizer_discriminator=optimizer_discriminator,
                            scheduler_generator=scheduler_generator,
                            scheduler_discriminator=scheduler_discriminator,
                            step=global_step,
                            loss=average_validation_loss,
                            best=new_best,
                            logger=logger,
                        )

        scheduler_discriminator.step()
        scheduler_generator.step()

        logger.info(
            f"train -- epoch: {epoch}, mel loss: {average_loss_mel:.4f}, generator loss: {average_loss_generator:.4f}, discriminator loss: {average_loss_discriminator:.4f}"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or finetune HiFi-GAN.")
    parser.add_argument(
        "dataset_dir",
        metavar="dataset-dir",
        help="path to the preprocessed data directory",
        type=Path,
    )
    parser.add_argument(
        "checkpoint_dir",
        metavar="checkpoint-dir",
        help="path to the checkpoint directory",
        type=Path,
    )
    parser.add_argument(
        "--resume",
        help="path to the checkpoint to resume from",
        type=Path,
    )
    parser.add_argument(
        "--finetune",
        help="whether to finetune (note that a resume path must be given)",
        action="store_true",
    )
    args = parser.parse_args()

    # display training setup info
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"CUDNN version: {torch.backends.cudnn.version()}")
    logger.info(f"CUDNN enabled: {torch.backends.cudnn.enabled}")
    logger.info(f"CUDNN deterministic: {torch.backends.cudnn.deterministic}")
    logger.info(f"CUDNN benchmark: {torch.backends.cudnn.benchmark}")
    logger.info(f"# of GPUS: {torch.cuda.device_count()}")

    # clear handlers
    logger.handlers.clear()

    world_size = torch.cuda.device_count()
    mp.spawn(
        train_model,
        args=(world_size, args),
        nprocs=world_size,
        join=True,
    )
