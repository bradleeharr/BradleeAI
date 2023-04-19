from torch.utils.data import DataLoader, TensorDataset, Dataset
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch
from torch.optim.lr_scheduler import StepLR
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
    def __init__(self, config, batch_size: int = 512):
        super().__init__()
        self.batch_size = batch_size
        self.plys = config.plys
        self.target_player = 'bubbakill7'
        self.num_layers = config.num_layers

    def setup(self, stage: str):
        # download_games_to_pgn(self.target_player)
        games = load_games_from_pgn(self.target_player + '_lichess_games.pgn')
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
                                                                                      test_set_size], \
                                                                          generator=torch.Generator().manual_seed(42))

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=16)  # input:(20,5), label:1

    def val_dataloader(self):
        return DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False, num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=16)


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class NeuralNetwork(nn.Module):
    def __init__(self, plys, num_hidden, num_layers, dropout=0.1):
        super(NeuralNetwork, self).__init__()
        input_size = 8 * 8 #* (plys + 1)  # Each board is 8 x 8 board with 12 input channels (6 white pieces, 6 black pieces)
        num_classes = 64 * 64  # From square * To square (4096 possible moves, although not all are valid)

        layers = []
        for i in range(num_layers):
            layers.append(torch.nn.Linear(num_hidden, num_hidden))
            layers.append(torch.nn.ReLU())
        self.hidden_layers = torch.nn.Sequential(*layers)

        self.fc1 = nn.Linear(input_size, num_hidden)
        self.fc3 = nn.Linear(num_hidden, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        x = F.relu(self.fc1(x))
        x = self.hidden_layers(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x


class ChessModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.net = NeuralNetwork(plys=config.plys, num_hidden=config.num_hidden, dropout=config.dropout,
                                 num_layers=config.num_layers)
        self.lr = config.lr
        self.gamma = config.gamma

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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = StepLR(optimizer, step_size=50, gamma=self.gamma)  # Adjust step_size and gamma as needed
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
