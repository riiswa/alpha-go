import os
import urllib
from typing import List
from urllib.error import HTTPError
from urllib.parse import urljoin

import requests
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from Goban import Board
from sgfmill import sgf
import torch
import numpy as np
from bs4 import BeautifulSoup

DIM = len(Board())


def download_games(urls: List[str]):
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Max-Age': '3600',
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'
    }

    sgf_files = []
    for url in urls:
        print(url)
        req = requests.get(url, headers)
        soup = BeautifulSoup(req.content, 'html.parser')

        if not os.path.exists("data/"):
            os.makedirs("data")

        for a in tqdm(soup.find_all('a')):
            sgf_file = a['href']
            if sgf_file.endswith('sgf'):
                # get data file from pickle
                clean_url = ''.join(filter(str.isalnum, url))
                try:
                    if not os.path.exists("data/" + clean_url + sgf_file):
                        urllib.request.urlretrieve(urljoin(url, sgf_file), "data/" + clean_url + sgf_file)
                    sgf_files.append("data/" + clean_url + sgf_file)
                except HTTPError as e:
                    print("Error 404 with: " + urljoin(url, sgf_file))
    return sgf_files


class SGFDataset(Dataset):
    color_dict = {'b': Board.BLACK, 'w': Board.WHITE}

    def __init__(self, sgf_files):
        self.sgf_files = sgf_files
        base = np.indices((DIM,))[0].reshape((Board._BOARDSIZE, Board._BOARDSIZE))
        self.transformations = []
        id = lambda x: x
        for i in [id, np.fliplr]:
            for j in [id, np.flipud]:
                for k in [id, np.rot90]:
                    self.transformations.append(i(j(k(base))))
        self.transformations = [t.flatten() for t in self.transformations]

    def __len__(self):
        return len(self.sgf_files)

    @staticmethod
    def get_color_layers_from_board(board: Board, color: str):
        b = board.board
        player = torch.zeros(DIM)
        opponent = torch.zeros(DIM)
        empty = torch.zeros(DIM)

        player[torch.where(torch.tensor(b == color))] = 1
        opponent[torch.where(torch.tensor(b == Board.flip(color)))] = 1
        empty[torch.where(torch.tensor(b == 0))] = 1

        return player, opponent, empty

    @staticmethod
    def get_liberties_layers_from_board(board: Board, n_layers: int):
        liberties = board.stringLiberties
        liberties_layers = []
        for i in range(n_layers):
            layer = torch.zeros(DIM)
            if i == (n_layers - 1):
                layer[torch.where(torch.tensor(liberties >= (i + 1)))] = 1
            else:
                layer[torch.where(torch.tensor(liberties == (i + 1)))] = 1
            liberties_layers.append(layer)
        return liberties_layers

    @staticmethod
    def get_legal_moves_layer(board):
        layer = torch.zeros(DIM)
        layer[board.legal_moves()] = 1
        return layer

    @staticmethod
    def get_chosen_move_layer(move):
        layer = torch.zeros(DIM)
        layer[move] = 1
        return layer

    def __getitem__(self, idx):

        with open(self.sgf_files[idx], "rb") as f:
            game = sgf.Sgf_game.from_bytes(f.read())
        board = Board()

        turn = 0
        winner = game.get_winner()
        for node in game.get_main_sequence():
            color, coords = node.get_move()
            if coords is not None:
                move = Board.flatten(coords)
                color_layers = list(self.get_color_layers_from_board(board, self.color_dict[color]))
                liberties_layers = self.get_liberties_layers_from_board(board, 4)
                legal_move_layer = self.get_legal_moves_layer(board)
                board.play_move(move)
                X = torch.stack(color_layers + liberties_layers + [legal_move_layer])

                move_layer = np.zeros(DIM)
                move_layer[move] = 1
                for transformation in self.transformations:
                    transformed_X = X[:, transformation].reshape((X.shape[0], 9, 9))
                    transformed_move_layer = move_layer[transformation].reshape((9, 9))
                    v = torch.tensor(color == winner)
                    yield transformed_X, transformed_move_layer, v
            turn += 1


if __name__ == "__main__":
    urls = [
        "https://homepages.cwi.nl/~aeb/go/games/games/other_sizes/9x9/Minigo/",
        "https://homepages.cwi.nl/~aeb/go/games/games/other_sizes/9x9/Go_Seigen/",
        "https://homepages.cwi.nl/~aeb/go/games/games/other_sizes/9x9/Misc/",
        "https://homepages.cwi.nl/~aeb/go/games/games/other_sizes/9x9/Misc/IgoFestival2008/index.html",
        "https://homepages.cwi.nl/~aeb/go/games/games/other_sizes/9x9/Misc/Iyama-6crown/index.html",
        "https://homepages.cwi.nl/~aeb/go/games/games/other_sizes/9x9/NHK/NewYear1990/index.html",
        "https://homepages.cwi.nl/~aeb/go/games/games/other_sizes/9x9/NHK/NewYear2002/index.html",
        "https://homepages.cwi.nl/~aeb/go/games/games/other_sizes/9x9/NHK/NewYear2003/index.html",
        "https://homepages.cwi.nl/~aeb/go/games/games/ProPairgo/2003/index.html",
        "https://homepages.cwi.nl/~aeb/go/games/games/ProPairgo/2004/index.html",
        "https://homepages.cwi.nl/~aeb/go/games/games/ProPairgo/2005/index.html",
        "https://homepages.cwi.nl/~aeb/go/games/games/ProPairgo/2006/index.html",
        "https://homepages.cwi.nl/~aeb/go/games/games/ProPairgo/2007/index.html",
        "https://homepages.cwi.nl/~aeb/go/games/games/ProPairgo/2008/index.html",
        "https://homepages.cwi.nl/~aeb/go/games/games/ProPairgo/2009/index.html",
        "https://homepages.cwi.nl/~aeb/go/games/games/other_sizes/9x9/computer/"
    ]
    dataset = SGFDataset(download_games(urls[:1]))
    for i in tqdm(range(len(dataset))):
        dataset[i]
    # dataloader = DataLoader(dataset, batch_size=32,
    #                         shuffle=True, num_workers=0)
    # for i_batch, sample_batched in enumerate(dataloader):
    #     print(sample_batched)
    #     break
    # data = []
    # i = 0
    # print(dataset[0])

