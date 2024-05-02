#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <sstream>
#include <cmath>

#include "mnist/mnist_reader.hpp"
#include "cifar10/cifar10_reader.hpp"

using namespace std;

typedef vector<double> vec_t;
typedef size_t label_t;

/*
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
conv2d (Conv2D)             (None, 32, 32, 8)         80        
                                                                
average_pooling2d (AverageP  (None, 8, 8, 8)          0         
ooling2D)                                                       
                                                                
flatten (Flatten)           (None, 512)               0         
                                                                
dense (Dense)               (None, 128)               65664     
                                                                
dropout (Dropout)           (None, 128)               0         
                                                                
dense_1 (Dense)             (None, 10)                1290      
                                                                
=================================================================
*/
struct struct_weights {
    vec_t conv_w;
    vec_t conv_b;
    vec_t dense_w;
    vec_t dense_b;
    vec_t dense1_w;
    vec_t dense1_b;
};

void read_from_file(vec_t &Indata, const string &FileName) {
    int size = Indata.size();
    ifstream source;
    source.open(FileName, ios::in);
    std::string line;
    for (int i = 0; i < size; i++) {
        std::getline(source, line, ',');
        std::istringstream in(line);
        in >> Indata[i];
        if (Indata[i] == 0.0) {
            std::cout << "Warning: value " << i << " is 0" << endl;
        }
    }
    std::cout << "Read size: " << size << endl;
    source.close();
}

void write_to_file(vec_t &data, const string &FileName) {
    int size = data.size();
    ofstream out(FileName, ios::out);
    std::string line;
    for (int i = 0; i < size; i++)
        out << data[i] << ",";
    std::cout << "Write size: " << size << endl;
    out.close();
}

auto LoadMNISTDataset(const string &datasetLocation) {
    auto dataset = mnist::read_dataset<std::vector, std::vector, double, uint8_t>(datasetLocation);
    cout << "Loaded MNIST test image size: " << dataset.test_images.size() << endl;
    return make_tuple(dataset.test_images, dataset.test_labels);
}

auto LoadCIFARDataset(const string &datasetLocation) {
    auto dataset = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(datasetLocation);
    cout << "Loaded CIFAR test image size: " << dataset.test_images.size() << endl;
    return make_tuple(dataset.test_images, dataset.test_labels);
}

vec_t infer_image(vec_t image, const struct_weights &weights) {

    auto begin = std::chrono::steady_clock::now();

    size_t channels = 1, height = 34, width = 34;

    // ------------------------------------------- conv2d ------------------------------------------------------
    // layers.Conv2D(8, (3, 3), strides=(1, 1), activation=square),
    size_t conv_channels = 8, kernel_height = 3, kernel_width = 3, stride = 1;
    if (((height - kernel_height) % stride != 0) || ((width - kernel_width) % stride != 0))
        throw std::invalid_argument("image size doesn't match conv parameters");
    size_t conv_height = (height - kernel_height) / stride + 1;
    size_t conv_width = (width - kernel_width) / stride + 1;
    vec_t conv_image(conv_channels * conv_height * conv_width);
    for (size_t c = 0; c < conv_channels; c++) {
        for (size_t h = 0; h < conv_height; h++) {
            for (size_t w = 0; w < conv_width; w++) {
                double sum = 0.0;
                for (size_t kh = 0; kh < kernel_height; kh++) {
                    for (size_t kw = 0; kw < kernel_width; kw++) {
                        sum += weights.conv_w[c * kernel_height * kernel_width + kh * kernel_width + kw] *
                               image[(h * stride + kh) * width + (w * stride + kw)];
                    }
                }
                conv_image[c * conv_height * conv_width + h * conv_width + w] = sum + weights.conv_b[c];
            }
        }
    }
    image = conv_image;
    channels = conv_channels;
    height = conv_height;
    width = conv_width;
    // square
    for (size_t j = 0; j < channels * height * width; j++)
        image[j] = image[j] * image[j];


    // ------------------------------------------- avg_pooling ------------------------------------------------------
    // layers.AveragePooling2D(pool_size=(4, 4)),
    size_t pool_height = 4, pool_width = 4;
    if ((height % pool_height != 0) || (width % pool_width != 0))
        throw std::invalid_argument("image size doesn't match avg_pool parameters");
    vec_t avgpool_image(conv_channels * (conv_height / pool_height) * (conv_width / pool_width));
    for (size_t c = 0; c < channels; c++) {
        for (size_t h = 0; h < height / pool_height; h++) {
            for (size_t w = 0; w < width / pool_width; w++) {
                double sum = 0.0;
                for (size_t ph = 0; ph < pool_height; ph++) {
                    for (size_t pw = 0; pw < pool_width; pw++) {
                        sum += image[c * height * width + (h * pool_height + ph) * width + (w * pool_width + pw)];
                    }
                }
                double avg = sum / static_cast<double>(pool_height * pool_width);
                avgpool_image[c * (height / pool_height) * (width / pool_width) + h * (width / pool_width) + w] = avg;
            }
        }
    }
    image = avgpool_image;
    height = height / pool_height;
    width = width / pool_width;


    // ------------------------------------------- dense ------------------------------------------------------
    // layers.Dense(128, activation=square),
    size_t dense_in_channels = channels * height * width;
    size_t dense_out_channels = 128;
    vec_t dense_image(dense_out_channels);

    for (size_t out = 0; out < dense_out_channels; out++) {
        double sum = 0.0;
        for (size_t in = 0; in < dense_in_channels; in++)
            sum += image[in] * weights.dense_w[out * dense_in_channels + in];
        dense_image[out] = sum + weights.dense_b[out];
    }
    image = dense_image;
    channels = dense_out_channels;
    height = 1;
    width = 1;
    // square
    for (size_t j = 0; j < channels * height * width; j++)
        image[j] = image[j] * image[j];


    // ------------------------------------------- dense1 ------------------------------------------------------
    // layers.Dense(10, activation=square),
    size_t dense1_in_channels = 128;
    size_t dense1_out_channels = 10;
    vec_t dense1_image(dense1_out_channels);
    for (size_t out = 0; out < dense1_out_channels; out++) {
        double sum = 0.0;
        for (size_t in = 0; in < dense1_in_channels; in++)
            sum += image[in] * weights.dense1_w[out * dense1_in_channels + in];
        dense1_image[out] = sum + weights.dense1_b[out];
    }

    auto end = std::chrono::steady_clock::now();
    auto t = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    // std::cout << "Time difference = " << t << " us" << std::endl;

    return dense1_image;
}

int main(int argc, char **argv) {
    string dataset;
    if (argc == 2) {
        dataset = argv[1];
    } else {
        cout << "Which dataset to test: ";
        cin >> dataset;
    }

    string dataset_path = "datasets/" + dataset;
    string weights_file_path = "weights/" + dataset;

    // (600, 120)
    // (120,)
    // (120, 84)
    // (84,)
    // (84, 10)
    // (10,)
    struct_weights weights = {
            vec_t(72), // (8, 1, 3, 3)
            vec_t(8),
            vec_t(65536), // (128, 8, 8, 8) first is out dense, second is channels, third is height, fourth is width
            vec_t(128),
            vec_t(1280), // (10, 128)
            vec_t(10),
    };
    read_from_file(weights.conv_w, weights_file_path + "/weights_conv_w.txt");
    read_from_file(weights.conv_b, weights_file_path + "/weights_conv_b.txt");
    read_from_file(weights.dense_w, weights_file_path + "/weights_dense_w.txt");
    read_from_file(weights.dense_b, weights_file_path + "/weights_dense_b.txt");
    read_from_file(weights.dense1_w, weights_file_path + "/weights_dense1_w.txt");
    read_from_file(weights.dense1_b, weights_file_path + "/weights_dense1_b.txt");

    int matchedCount = 0;
    // long timeUsed = 0;

    // load dataset
    if (dataset == "mnist" || dataset == "fmnist") {
        auto [testImages, testLabels] = LoadMNISTDataset(dataset_path);
        for (size_t index = 0; index < testImages.size(); index++) {
            auto image = testImages[index];
            size_t channels = 1;
            size_t height = 28;
            size_t width = 28;

            // zero-pad
            // mnist/fmnist (3, 3) and (3, 3)
            size_t left_pad = 3, right_pad = 3, top_pad = 3, bottom_pad = 3;
            size_t zeropad_height = height + top_pad + bottom_pad;
            size_t zeropad_width = width + left_pad + right_pad;
            vec_t zeropad_image(zeropad_height * zeropad_width, 0.0);
            for (size_t h = 0; h < height; h++)
                for (size_t w = 0; w < width; w++)
                    zeropad_image[(h + top_pad) * zeropad_width + (w + left_pad)] = image[h * width + w] / 255.0;

            // timeUsed += t;
            auto result_image = infer_image(zeropad_image, weights);

            // softmax
            double sum = 0.0;
            for (size_t c = 0; c < 10; c++) {
                result_image[c] = exp(result_image[c]);
                sum += result_image[c];
            }
            for (size_t c = 0; c < 10; c++)
                result_image[c] = result_image[c] / sum;

            auto predicted = std::distance(std::begin(result_image),
                                           std::max_element(std::begin(result_image), std::end(result_image)));
            bool correct = (testLabels[index] == predicted);
            if (!correct) cout << index << ", " << flush;
            matchedCount += correct;
        }
        // cout << "Total time: " << timeUsed << "us" << endl;
        cout << "Matched count: " << matchedCount << "/" << testImages.size() << endl;
        cout << "Accuracy: " << (double) matchedCount / (double) testImages.size() << endl;
    } else if (dataset == "cifar10" || dataset == "cifar100") {
        auto [testImages, testLabels] = LoadCIFARDataset(dataset_path);
        for (size_t index = 0; index < testImages.size(); index++) {
            auto uint8_image = testImages[index];
            size_t channels = 3;
            size_t height = 32;
            size_t width = 32;

            // 3in1
            vec_t new_image(1 * height * width);
            for (size_t h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    double sum = 0.0;
                    for (size_t c = 0; c < channels; c++) {
                        sum += uint8_image[c * height * width + h * width + w] / 255.0;
                    }
                    double avg = sum / channels;
                    new_image[h * width + w] = avg;
                }
            }

            // zero-pad
            // cifar10/cifar100 (1, 1) and (1, 1)
            size_t left_pad = 1, right_pad = 1, top_pad = 1, bottom_pad = 1;
            size_t zeropad_height = height + top_pad + bottom_pad;
            size_t zeropad_width = width + left_pad + right_pad;
            vec_t zeropad_image(zeropad_height * zeropad_width, 0.0);
            for (size_t h = 0; h < height; h++)
                for (size_t w = 0; w < width; w++)
                    zeropad_image[(h + top_pad) * zeropad_width + (w + left_pad)] = new_image[h * width + w];

            // timeUsed += t;
            auto result_image = infer_image(zeropad_image, weights);

            // softmax
            double sum = 0.0;
            for (size_t c = 0; c < 10; c++) {
                result_image[c] = exp(result_image[c]);
                sum += result_image[c];
            }
            for (size_t c = 0; c < 10; c++)
                result_image[c] = result_image[c] / sum;

            auto predicted = std::distance(std::begin(result_image),
                                           std::max_element(std::begin(result_image), std::end(result_image)));
            bool correct = (testLabels[index] == predicted);
            if (!correct) cout << index << ", " << flush;
            matchedCount += correct;
        }
        // cout << "Total time: " << timeUsed << "us" << endl;
        cout << "Matched count: " << matchedCount << "/" << testImages.size() << endl;
        cout << "Accuracy: " << (double) matchedCount / (double) testImages.size() << endl;
    }
    return 0;
}
