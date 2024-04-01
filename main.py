import argparse
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DeepSpeedStrategy

from data_module import TrainingDataModule
from model import MCLLaVAModel
from wsd_scheduler import WSDParameters

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
        "-lrm",
        "--learning_rate_min",
        type=float,
        dest="learning_rate_min",
        default=0.1,
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
        default=0.03,
    )
    parser.add_argument(
        "-sr",
        "--stable_ratio",
        type=float,
        dest="stable_ratio",
        default=0.5,
    )
    parser.add_argument(
        "-ar",
        "--annealing_ratio",
        type=float,
        dest="annealing_ratio",
        default=0.1,
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
    parser.add_argument(
        "-ab",
        "--accumulate_batches",
        type=int,
        dest="accumulate_batches",
    )
    args = parser.parse_args()

    data_module = TrainingDataModule(args.batch_size, args.data_dir)

    train_steps = int(
        len(data_module.dataset)
        / args.batch_size
        * args.epochs_num
        / torch.cuda.device_count()
        / args.accumulate_batches
    )
    scheduler_params = WSDParameters(
        total_steps=train_steps,
        warmup_ratio=args.warmup_ratio,
        stable_ratio=args.stable_ratio,
        annealing_ratio=args.annealing_ratio,
        min_lr_ratio=args.learning_rate_min,
    )

    if args.checkpoint_path != "":
        model = MCLLaVAModel.load_from_checkpoint(
            checkpoint_path=args.checkpoint_path,
            lr=args.learning_rate,
            freeze_vision=args.freeze_vision,
            freeze_text=args.freeze_text,
            scheduler_params=scheduler_params,
            use_lora=args.use_lora,
            bits=args.bits,
        )
    else:
        model = MCLLaVAModel(
            lr=args.learning_rate,
            freeze_vision=args.freeze_vision,
            freeze_text=args.freeze_text,
            scheduler_params=scheduler_params,
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
        strategy=DeepSpeedStrategy(stage=2),
        accumulate_grad_batches=args.accumulate_batches,
    )
    params = {
        "batch_size": data_module.batch_size,
        "epochs_num": args.epochs_num,
        "learning_rate": args.learning_rate,
    }
    wandb_logger.log_hyperparams(params)
    trainer.fit(model=model, datamodule=data_module)
    trainer.save_checkpoint(str(log_dir / "model.ckpt"))
