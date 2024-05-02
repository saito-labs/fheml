import argparse
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# Defining the convolutional neural network
class XNet10(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=3)
        self.avg_pool = nn.AvgPool2d(4)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv(x)
        x = pow(x, 2)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = pow(x, 2)
        x = self.dropout(x)
        x = self.fc2(x)
        return pow(x, 2)


def train(args):
    if args.dataset == "mnist":
        train_dataset = datasets.MNIST(root="datasets", train=True, download=True, transform=ToTensor())
        test_dataset = datasets.MNIST(root="datasets", train=False, download=True, transform=ToTensor())
    elif args.dataset == "fmnist":
        train_dataset = datasets.FashionMNIST(root="datasets", train=True, download=True, transform=ToTensor())
        test_dataset = datasets.FashionMNIST(root="datasets", train=False, download=True, transform=ToTensor())
    elif args.dataset == "cifar10":
        train_dataset = datasets.CIFAR10(root="datasets", train=True, download=True, transform=ToTensor())
        test_dataset = datasets.CIFAR10(root="datasets", train=False, download=True, transform=ToTensor())
    else:
        raise NameError

    # if args.dataset in ["mnist", "fmnist"]:
    #     print(x_train.shape)
    #     N_train, w, h, = x_train.shape
    #     N_test, _, _ = x_test.shape
    #     x_train_new = np.zeros([N_train, h + 6, w + 6])
    #     x_test_new = np.zeros([N_test, h + 6, w + 6])
    #     x_train_new[0:N_train, 3:h + 3, 3:w + 3] = x_train[:, :, :]
    #     x_test_new[0:N_test, 3:h + 3, 3:w + 3] = x_test[:, :, :]
    # elif args.dataset in ["cifar10", "cifar100"]:
    #     x_train_new = np.average(x_train, axis=3, keepdims=False)
    #     x_test_new = np.average(x_test, axis=3, keepdims=False)
    #     x_train = x_train_new
    #     x_test = x_test_new
    #
    #     N_train, w, h, = x_train.shape
    #     N_test, _, _ = x_test.shape
    #     x_train_new = np.zeros([N_train, h + 2, w + 2])
    #     x_test_new = np.zeros([N_test, h + 2, w + 2])
    #     x_train_new[0:N_train, 1:h + 1, 1:w + 1] = x_train[:, :, :]
    #     x_test_new[0:N_test, 1:h + 1, 1:w + 1] = x_test[:, :, :]
    # else:
    #     raise ValueError("No Such Dataset")
    #
    # x_train = x_train_new
    # x_test = x_test_new
    #
    # # channel last
    # input_shape = (34, 34, 1)
    # x_train = np.expand_dims(x_train, -1)
    # x_test = np.expand_dims(x_test, -1)
    # print("x_train shape:", x_train.shape)
    #
    # print(x_train.shape[0], "train samples")
    # print(x_test.shape[0], "test samples")
    #
    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size * 10, shuffle=False)

    model = XNet10().to(device)
    print(f"Using {device} device")
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train the model
    loss_list = []
    for epoch in range(args.num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        loss_list.append(running_loss)
        print('Epoch %d loss: %.3f' % (epoch + 1, running_loss))

    # Test
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy: %.2f %%' % (100 * correct / total))

    torch.save(model.state_dict(), 'model_weights.pth')
    # model.load_state_dict(torch.load('model_weights.pth'))
    # model.eval()


    # checkpoint_filepath = '/tmp/checkpoint'
    # model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    #     filepath=checkpoint_filepath,
    #     save_weights_only=True,
    #     monitor='val_accuracy',
    #     mode='max',
    #     save_best_only=True)
    #
    # model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_split=0.1,
    #           callbacks=[model_checkpoint_callback])
    #
    # model.load_weights(checkpoint_filepath)
    #
    # score = model.evaluate(x_test, y_test)
    # print("Test loss:", score[0])
    # print("Test accuracy:", score[1])
    #
    # weights_list = model.get_weights()
    # print(len(weights_list))
    # weights_list[0] = weights_list[0].transpose((3, 2, 0, 1))
    # weights_list[2] = weights_list[2].reshape((8, 8, 8, 128)).transpose(
    #     (3, 2, 0, 1))  # first is height, second is width, third is channel, fourth is output
    # weights_list[4] = weights_list[4].transpose((1, 0))
    # for weights in weights_list:
    #     print(weights.shape)
    #
    # np.savetxt(f"./weights/{args.dataset}/weights_conv_w.txt", weights_list[0].flatten(), fmt="%f", delimiter=",",
    #            newline=",")
    # np.savetxt(f"./weights/{args.dataset}/weights_conv_b.txt", weights_list[1].flatten(), fmt="%f", delimiter=",",
    #            newline=",")
    # np.savetxt(f"./weights/{args.dataset}/weights_dense_w.txt", weights_list[2].flatten(), fmt="%f", delimiter=",",
    #            newline=",")
    # np.savetxt(f"./weights/{args.dataset}/weights_dense_b.txt", weights_list[3].flatten(), fmt="%f", delimiter=",",
    #            newline=",")
    # np.savetxt(f"./weights/{args.dataset}/weights_dense1_w.txt", weights_list[4].flatten(), fmt="%f", delimiter=",",
    #            newline=",")
    # np.savetxt(f"./weights/{args.dataset}/weights_dense1_b.txt", weights_list[5].flatten(), fmt="%f", delimiter=",",
    #            newline=",")
    #
    # model.save_weights(f"./weights/{args.dataset}/tf-weights_{args.dataset}.h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process configs")
    parser.add_argument("--dataset", type=str, default="mnist", help="dataset name")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    arguments = parser.parse_args()
    train(arguments)
