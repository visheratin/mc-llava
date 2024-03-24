import argparse
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DeepSpeedStrategy

from data_module import TrainingDataModule
from model import MCLLaVAModel

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        dest="learning_rate",
    )
    parser.add_argument(
        "-vlr",
        "--vision_learning_rate",
        type=float,
        dest="vision_learning_rate",
    )
    parser.add_argument(
        "-tlr",
        "--text_learning_rate",
        type=float,
        dest="text_learning_rate",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        dest="batch_size",
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        dest="data_dir",
    )
    parser.add_argument(
        "-n",
        "--epochs_num",
        type=int,
        help="number of epochs to train",
        dest="epochs_num",
    )
    parser.add_argument(
        "-cp",
        "--checkpoint_path",
        type=str,
        dest="checkpoint_path",
        default="",
    )
    parser.add_argument(
        "-wr",
        "--warmup_ratio",
        type=float,
        dest="warmup_ratio",
    )
    parser.add_argument(
        "-l",
        "--use_lora",
        type=bool,
        dest="use_lora",
    )
    parser.add_argument(
        "-fv",
        "--freeze_vision",
        type=bool,
        dest="freeze_vision",
    )
    parser.add_argument(
        "-ft",
        "--freeze_text",
        type=bool,
        dest="freeze_text",
    )
    parser.add_argument(
        "-b",
        "--bits",
        type=int,
        dest="bits",
    )
    args = parser.parse_args()

    data_module = TrainingDataModule(args.batch_size, args.data_dir)

    train_steps = int(len(data_module.dataset) / args.batch_size * args.epochs_num / torch.cuda.device_count())
    warmup_steps = int(train_steps * args.warmup_ratio)

    if args.checkpoint_path != "":
        model = MCLLaVAModel.load_from_checkpoint(
            checkpoint_path=args.checkpoint_path,
            lr=args.learning_rate,
            freeze_vision=args.freeze_vision,
            freeze_text=args.freeze_text,
            total_steps=train_steps,
            warmup_steps=warmup_steps,
            use_lora=args.use_lora,
            bits=args.bits,
        )
    else:
        model = MCLLaVAModel(
            lr=args.learning_rate,
            freeze_vision=args.freeze_vision,
            freeze_text=args.freeze_text,
            total_steps=train_steps,
            warmup_steps=warmup_steps,
            use_lora=args.use_lora,
            bits=args.bits,
        )

    log_dir = Path("training") / "logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    wandb_logger = WandbLogger(
        project="mc-llava",
        save_dir=str(log_dir),
        job_type="train",
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs_num,
        accelerator="gpu",
        devices=-1,
        precision="bf16-mixed",
        log_every_n_steps=1,
        logger=wandb_logger,
        strategy=DeepSpeedStrategy(stage=3),
    )
    params = {
        "batch_size": data_module.batch_size,
        "epochs_num": args.epochs_num,
        "learning_rate": args.learning_rate,
    }
    wandb_logger.log_hyperparams(params)
    trainer.fit(model=model, datamodule=data_module)
    trainer.save_checkpoint(str(log_dir / "model.ckpt"))
