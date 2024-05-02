# Python for training CIFAR-10, MNIST, FMNIST  in LeNet-5
import argparse
import numpy as np
from tensorflow import keras

def square(x):
    return pow(x, 2)

def train(args):
    if args.dataset == "mnist":
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        num_classes = 10
    elif args.dataset == "fmnist":
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        num_classes = 10
    elif args.dataset == "cifar10":
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        num_classes = 10
    elif args.dataset == "cifar100":
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
        num_classes = 100
    else:
        exit(-1)
    
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    if args.dataset in ["mnist", "fmnist"]:
        print(x_train.shape)
        N_train, w , h, = x_train.shape
        N_test, _ , _ = x_test.shape
        x_train_new = np.zeros([N_train, h+6, w+6])
        x_test_new = np.zeros([N_test, h+6, w+6])
        x_train_new[0:N_train, 3:h+3, 3:w+3] = x_train[:,:,:]
        x_test_new[0:N_test, 3:h+3, 3:w+3] = x_test[:,:,:]
    elif args.dataset in ["cifar10", "cifar100"]:
        x_train_new = np.average(x_train, axis=3, keepdims=False)
        x_test_new = np.average(x_test, axis=3, keepdims=False)
        x_train = x_train_new
        x_test = x_test_new

        N_train, w , h, = x_train.shape
        N_test, _ , _ = x_test.shape
        x_train_new = np.zeros([N_train, h+2, w+2])
        x_test_new = np.zeros([N_test, h+2, w+2])
        x_train_new[0:N_train, 1:h+1, 1:w+1] = x_train[:,:,:]
        x_test_new[0:N_test, 1:h+1, 1:w+1] = x_test[:,:,:]
    else:
        raise ValueError("No Such Dataset")

    x_train = x_train_new
    x_test = x_test_new

    # channel last
    input_shape = (34, 34, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)

    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            keras.layers.Conv2D(8, (3, 3), strides=(1, 1), activation=square),
            keras.layers.AveragePooling2D(pool_size=(4, 4)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation=square),
            keras.layers.Dropout(0.5),
            # keras.layers.Dense(84, activation=square),
            # keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation='softmax'),
        ]
    )

    model.summary()

    batch_size = 128
    epochs = 200

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    checkpoint_filepath = '/tmp/checkpoint'
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[model_checkpoint_callback])

    model.load_weights(checkpoint_filepath)

    score = model.evaluate(x_test, y_test)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    weights_list = model.get_weights()
    print(len(weights_list))
    weights_list[0] = weights_list[0].transpose((3, 2, 0, 1))
    weights_list[2] = weights_list[2].reshape((8, 8, 8, 128)).transpose((3, 2, 0, 1)) # first is height, second is width, third is channel, fourth is output
    weights_list[4] = weights_list[4].transpose((1, 0))
    for weights in weights_list:
        print(weights.shape)
    
    np.savetxt(f"./weights/{args.dataset}/weights_conv_w.txt", weights_list[0].flatten(), fmt="%f", delimiter=",", newline=",")
    np.savetxt(f"./weights/{args.dataset}/weights_conv_b.txt", weights_list[1].flatten(), fmt="%f", delimiter=",", newline=",")
    np.savetxt(f"./weights/{args.dataset}/weights_dense_w.txt", weights_list[2].flatten(), fmt="%f", delimiter=",", newline=",")
    np.savetxt(f"./weights/{args.dataset}/weights_dense_b.txt", weights_list[3].flatten(), fmt="%f", delimiter=",", newline=",")
    np.savetxt(f"./weights/{args.dataset}/weights_dense1_w.txt", weights_list[4].flatten(), fmt="%f", delimiter=",", newline=",")
    np.savetxt(f"./weights/{args.dataset}/weights_dense1_b.txt", weights_list[5].flatten(), fmt="%f", delimiter=",", newline=",")
    
    model.save_weights(f"./weights/{args.dataset}/tf-weights_{args.dataset}.h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process configs")
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset name")
    args = parser.parse_args()
    print(args.dataset)
    train(args)
