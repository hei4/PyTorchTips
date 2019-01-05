# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data


def main():
    x1 = torch.rand(100, 3)
    y1 = torch.rand(100, 1)

    # x1とy1からデータセット1を作成
    dataset1 = torch.utils.data.TensorDataset(x1, y1)
    print(len(dataset1))

    # データセット1の最初のレコード
    print(dataset1[0])

    x2 = torch.rand(200, 3)
    y2 = torch.rand(200, 1)

    # x2とy2からデータセット2を作成
    dataset2 = torch.utils.data.TensorDataset(x2, y2)
    print(len(dataset2))

    # データセット1とデータセット2を結合してデータセット3を作成
    dataset3 = torch.utils.data.ConcatDataset([dataset1, dataset2])
    print(len(dataset3))


    dataset4 = torch.utils.data.Subset(dataset3, [i for i in range(200, 300)])
    print(len(dataset4))


if __name__ == '__main__':
    main()