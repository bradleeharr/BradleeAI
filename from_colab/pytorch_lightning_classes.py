import logging

from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.nn.init as init
import torch
import random

from torch.optim.lr_scheduler import StepLR, CyclicLR, CosineAnnealingLR

from from_colab.neural_network_model import NeuralNetwork
from utilities import *


class ChessDataset(Dataset):
    def __init__(self, X, Y, plys):
        self.X = X
        self.Y = Y
        self.plys = plys  # plys ahead for prediction

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        x = torch.tensor(self.X[index, :, :]).float()
        y = torch.tensor(self.Y[index]).long()
        # board = input_to_board(x, self.plys)
        # move = flat_to_move(y, board)
        # print(board)
        # print(move)
        return x, y


class ChessDataModule(pl.LightningDataModule):
    def __init__(self, config, batch_size: int = 512, target_player='bubbakill7'):
        super().__init__()
        self.batch_size = batch_size
        self.plys = config.plys
        self.target_player = target_player

    def setup(self, stage: str):
        # download_games_to_pgn(self.target_player)
        print("Loading games for player: " + self.target_player)
        games = load_games_from_pgn(self.target_player + '_lichess_games.pgn')
        print("Loaded games.")
        allX = []
        ally = []
        idx = 0
        for game in games:
            positions, target_moves = process_game(game, self.target_player, num_previous_positions=self.plys)
            allX.extend(positions)
            ally.extend(target_moves)
            idx = idx + 1
            if idx % 100 == 0: print('\rPreprocessing Game ' + str(idx), end=" ")

        allX = np.array(allX)
        ally = np.array(ally)

        chessData = ChessDataset(allX, ally, plys=self.plys)
        train_set_size = int(len(chessData) * 0.7)
        valid_set_size = int(len(chessData) * 0.15)
        test_set_size = len(chessData) - train_set_size - valid_set_size
        self.train_set, self.valid_set, self.test_set = data.random_split(chessData, [train_set_size, valid_set_size,
                                                                                      test_set_size],
                                                                          generator=torch.Generator().manual_seed(42))

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)


class ChessModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.net = NeuralNetwork(plys=config.plys, hidden_layer_size=config.hidden_layer_size, dropout=config.dropout,
                                 num_conv_blocks=config.num_conv_blocks,
                                 pooling_interval=config.pooling_interval, squeeze_factor=config.squeeze_factor)
        self.lr = config.lr
        self.gamma = config.gamma
        self.lr_step_size = config.lr_step_size
        self.weight_decay = config.weight_decay
        self.scheduler_type = config.scheduler_type
        self.momentum = config.momentum

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.net(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.net(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('validation_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        y_hat = self.net(x)
        test_loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", test_loss)
        try:
            boards = []
            for board in x:
                boards.append(input_to_board(board))

            legal_moves_mask = get_legal_moves_mask(boards)
            legal_moves_mask_tensor = torch.tensor(legal_moves_mask, dtype=torch.bool, device=x.device)
            count = torch.count_nonzero(legal_moves_mask_tensor)
            y_hat_filtered = filter_illegal_moves(y_hat, legal_moves_mask_tensor)

            _, predicted_move_indices = torch.max(y_hat_filtered, dim=1)
            accuracy = torch.mean((predicted_move_indices == y).float())

            random_move_indices = torch.zeros_like(y)
            for idx, legal_moves in enumerate(legal_moves_mask):
                legal_moves_list = [move_idx for move_idx, is_legal in enumerate(legal_moves) if is_legal]
                random_move = random.choice(legal_moves_list)
                random_move_indices[idx] = random_move

            random_accuracy = torch.mean((random_move_indices == y).float())

            for idx, move_index in enumerate(predicted_move_indices):
                print("predicted test move: ", flat_to_move(move_index, boards[idx]), end=' ')
                print("actual test move: " + str(flat_to_move(y[idx], boards[idx])) + ' On board \n' + str(boards[idx]))
            self.log("move_prediction_accuracy", accuracy)
            self.log("random_move_prediction_accuracy", random_accuracy)

        except Exception as e:
            logging.error('error happened during move prediction', exc_info=e)

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                                    momentum=self.momentum)
        if self.scheduler_type == 'steplr':
            scheduler = StepLR(optimizer, step_size=self.lr_step_size, gamma=self.gamma)
        elif self.scheduler_type == 'cycliclr':
            scheduler = CyclicLR(optimizer, base_lr=self.lr / 10, max_lr=self.lr,
                                 step_size_up=self.lr_step_size, mode='triangular2')
        elif self.scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.lr_step_size,
                                          eta_min=self.lr / 10)
        else:
            raise ValueError(f"Invalid scheduler_type: {self.scheduler_type}")

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
