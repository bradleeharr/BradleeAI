from neural_network_model import *
from pytorch_lightning_classes import *
from utilities import *
import os
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor


def train_model(target_player='bubbakill7'):
    # Get the Data
    my_username = target_player
    num_games = 10000
    batch_size = 16
    if not os.path.exists(my_username + '_lichess_games.pgn'):
        download_games_to_pgn(my_username, num_games)
    else:
        print("Games found")

    # Create the early stopping callback
    early_stopping_callback = EarlyStopping(monitor="validation_loss", min_delta=0.003, patience=5, verbose=True,
                                            mode="min")

    # Create the learning rate monitor callback
    lr_monitor = LearningRateMonitor(logging_interval="step")

    wandb.init(project="chess_testing_sweep")
    config = wandb.config
    wandb_logger = WandbLogger()
    data = ChessDataModule(config=config, batch_size=batch_size, target_player=target_player)
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
    trainer = pl.Trainer(accelerator=accelerator, devices=devices, max_epochs=10000,
                         default_root_dir="./lightning-example", logger=wandb_logger,
                         callbacks=[early_stopping_callback, lr_monitor])
    print("Beginning training fit")
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
            'dropout': {'max': 0.8, 'min': 0.0},
            'hidden_layer_size': {'max': 1000, 'min': 10},
            'plys': {'max': 16, 'min': 1},  # {'max': 24, 'min': 1},
            'lr': {'distribution': 'log_uniform_values', 'max': 2., 'min': 1e-9},
            'momentum': {'distribution': 'log_uniform_values', 'max': 2., 'min': 1e-9},
            'gamma': {'max': 1.1, 'min': 1e-4},
            'num_conv_blocks': {'value': 3},
            'lr_step_size': {'max': 2000, 'min': 1},
            'pooling_interval': {'max': 6, 'min': 1},
            'weight_decay': {'max': 0.7, 'min': 0.0},
            'scheduler_type': {'values': ['steplr', 'cycliclr', 'cosine']},
            'squeeze_factor': {'values': [2, 3, 4, 5, 6, 7, 8]},

        }
    }

    sweep_id = wandb.sweep(sweep_config, project="openings_removed_chess_7_layer_residual_sweep3")
    wandb.agent(sweep_id=sweep_id, function=train_model, count=100)
