# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

import torch.nn as nn
import torch.optim as optim
import torch.utils.data


def main():
    num_workers = 2
    batch_size = 50
    epoch_size = 20

    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None,
                     names=['sepal-length',
                            'sepal-width',
                            'petal-length',
                            'petal-width',
                            'class'])

    class_mapping = {label:idx for idx, label in enumerate(np.unique(df['class']))}
    df['class'] = df['class'].map(class_mapping)

    features = torch.tensor(df[['sepal-length', 'sepal-width', 'petal-length', 'petal-width']].values,
                            dtype=torch.float)

    labels = torch.tensor(df['class'].values, dtype=torch.long)

    train_set = torch.utils.data.TensorDataset(features, labels)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    net = torch.nn.Sequential(
        nn.Linear(4, 64),
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 4)
    )
    print(net)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    net.to(device)  # for GPU

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    epoch_list = []
    train_acc_list = []
    # test_acc_list = []
    for epoch in range(epoch_size):

        train_true = []
        train_pred = []
        for itr, data in enumerate(train_loader):
            features, labels = data
            train_true.extend(labels.tolist())

            features, labels = features.to(device), labels.to(device)   # for GPU

            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            train_pred.extend(predicted.tolist())

            print('[epochs: {}, mini-batches: {}, records: {}] loss: {:.3f}'.format(
                epoch + 1, itr + 1, (itr + 1) * batch_size, loss.item()))

        train_acc = accuracy_score(train_true, train_pred)
        # test_acc = accuracy_score(test_true, test_pred)
        # print('    epocs: {}, train acc.: {:.3f}, test acc.: {:.3f}'.format(epoch + 1, train_acc, test_acc))
        print('    epocs: {}, train acc.: {:.3f}'.format(epoch + 1, train_acc))
        print()

        epoch_list.append(epoch + 1)
        train_acc_list.append(train_acc)
        #test_acc_list.append(test_acc)

    print('Finished Training')

    print('Save Network')
    torch.save(net.state_dict(), 'model.pth')

    # df = pd.DataFrame({'epoch': epoch_list,
    #                    'train/accuracy': train_acc_list,
    #                    'test/accuracy': test_acc_list})
    df = pd.DataFrame({'epoch': epoch_list,
                       'train/accuracy': train_acc_list})

    print('Save Training Log')
    df.to_csv('train.log', index=False)


if __name__ == '__main__':
    main()
