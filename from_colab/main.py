from neural_network_model import *
from pytorch_lightning_classes import *
from utilities import *
import os
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor


def train_model():
    # Get the Data
    my_username = 'bubbakill7'
    num_games = 100
    batch_size = 8
    if not os.path.exists(my_username + '_lichess_games.pgn'):
        download_games_to_pgn(my_username, num_games)
    else:
        print("Games found")

    # Create the early stopping callback
    early_stopping_callback = EarlyStopping(monitor="validation_loss", min_delta=0.001, patience=15, verbose=True,
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
            'dropout': {'max': 0.8, 'min': 0.0},
            'hidden_layer_size': {'values': [0, 0]},
            'plys': {'max': 24, 'min': 0},
            'num_hidden_layers': {'values': [0, 0]},
            'lr': {'distribution': 'log_uniform_values', 'max': 2., 'min': 1e-6},
            'gamma': {'max': 1.2, 'min': 1e-4},
            'num_conv_blocks': {'max': 12, 'min': 2},
            'lr_step_size': {'max': 200, 'min': 10},
            'pooling_interval': {'max': 6, 'min': 1},
            'weight_decay': {'max': 0.2, 'min': 0.0},
            'scheduler_type': {'values': ['steplr', 'cycliclr', 'cosine']},
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="chess_sweep_with_plys")
    wandb.agent(sweep_id=sweep_id, function=train_model, count=50)
