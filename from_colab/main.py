from neural_network_model import *
from utilities import *
import os
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor


def train_model():
    # Get the Data
    my_username = 'bubbakill7'
    num_games = 1
    batch_size = 128
    if not os.path.exists(my_username+'_lichess_games.pgn'):
        download_games_to_pgn(my_username, num_games)
    else:
        print("Games found")

    # Create the early stopping callback
    early_stopping_callback = EarlyStopping(monitor="validation_loss", min_delta=0.001, patience=550, verbose=True,
                                            mode="min")

    # Create the learning rate monitor callback
    lr_monitor = LearningRateMonitor(logging_interval="step")

    wandb.init(project="chess_testing_sweep")
    config = wandb.config
    wandb_logger = WandbLogger()
    data = ChessDataModule(config=config, batch_size=batch_size)
    model = ChessModel(config=config)

    wandb_logger.watch(model.net)  # show gradient

    # Check if a GPU is available
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = 1
    else:
        accelerator = "cpu"
        devices = 1

    # Create the PyTorch Lightning Trainer and train your model
    trainer = pl.Trainer(accelerator=accelerator, devices=devices, max_epochs=3000,
                         default_root_dir="./lightning-example", logger=wandb_logger,
                         callbacks=[early_stopping_callback, lr_monitor])
    trainer.fit(model, data)
    trainer.test(model, data)


if __name__ == '__main__':
    sweep_config = {
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 2
        },
        'method': 'bayes',
        'name': 'first_sweep',
        'metric': {
            'goal': 'minimize',
            'name': 'validation_loss'
        },
        'parameters': {
            'num_hidden': {'max': 150, 'min': 3},
            'lr': {'distribution': 'log_uniform_values', 'max': 2., 'min': 1e-9},
            'dropout': {'values': [0, 0]},
            'plys': {'values': [0, 0, 0]},
            'gamma': {'max': 1.5, 'min': 0.3},
            'num_layers': {'max': 15, 'min': 0}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="chess_sweep_8")
    wandb.agent(sweep_id=sweep_id, function=train_model, count=500)