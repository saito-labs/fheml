#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <sstream>
#include <cmath>
#include <cassert>

#include "mnist/mnist_reader.hpp"
#include "cifar10/cifar10_reader.hpp"

#include "CAHEL/cahel.h"

using namespace std;

typedef vector<double> vec_t;
typedef size_t label_t;

//double scale = std::pow(2.0, 40);

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

struct struct_encoded_weights {
    vector<CAHELGPUPlaintext> conv_w;
    CAHELGPUPlaintext conv_b;
    CAHELGPUPlaintext pool_mask;
    vector<CAHELGPUPlaintext> dense_w;
    CAHELGPUPlaintext dense_mask;
    CAHELGPUPlaintext dense_b;
    CAHELGPUPlaintext dense1_w;
    CAHELGPUPlaintext dense1_b;
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

auto encode_weights(const CAHELGPUContext &context, CAHELGPUCKKSEncoder &encoder, const struct_weights &weights) {
    struct_encoded_weights encoded_weights;
    double scale = std::pow(2.0, 40);

    //------------------------------conv_w-----------------------------
    vector<vec_t> encoded_conv_w(9);

    for (size_t i = 0; i < 9; i++) {
        for (size_t c = 0; c < 8; c++) {
            for (size_t j = 0; j < 1024; j++) {
                encoded_conv_w[i].push_back(weights.conv_w[c * 9 + i]);
            }
        }
    }

    for (auto &w: encoded_conv_w) {
        CAHELGPUPlaintext pt;
        encoder.encode(context, w, scale, pt);
        encoded_weights.conv_w.push_back(pt);
    }
    cout << "conv_w encoded" << endl;


    //------------------------------conv_b-----------------------------
    vec_t encoded_conv_b(8192, 0.0);

    for (size_t c = 0; c < 8; c++) {
        for (size_t j = 0; j < 1024; j++) {
            encoded_conv_b[c * 1024 + j] = weights.conv_b[c];
        }
    }

    {
        CAHELGPUPlaintext pt;
        encoder.encode(context, encoded_conv_b, scale, pt);
        mod_switch_to_next_inplace(context, pt);
        encoded_weights.conv_b = pt;
    }
    cout << "conv_b encoded" << endl;


    //------------------------------pool_mask-----------------------------
    vec_t encoded_pool_mask(8192, 0.0);

    for (size_t i = 0; i < 512; i++) {
        encoded_pool_mask[i * 16] = 1.0 / 16.0;
    }

    {
        CAHELGPUPlaintext pt;
        encoder.encode(context, encoded_pool_mask, scale, pt);
        mod_switch_to_next_inplace(context, pt);
        mod_switch_to_next_inplace(context, pt);
        encoded_weights.pool_mask = pt;
    }
    cout << "pool_mask encoded" << endl;


    //------------------------------dense_w-----------------------------
    vector<vec_t> encoded_dense_w(8);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 512; j++) {
            for (size_t k = 0; k < 16; k++) {
                encoded_dense_w[i].push_back(weights.dense_w[(16 * i + k) * 512 + j]);
            }
        }
    }

    for (auto &w: encoded_dense_w) {
        CAHELGPUPlaintext pt;
        encoder.encode(context, w, scale, pt);
        mod_switch_to_next_inplace(context, pt);
        mod_switch_to_next_inplace(context, pt);
        mod_switch_to_next_inplace(context, pt);
        encoded_weights.dense_w.push_back(pt);
    }
    cout << "dense_w encoded" << endl;


    //------------------------------dense_mask-----------------------------
    vec_t encoded_dense_mask(8192, 0.0);
    for (size_t i = 0; i < 16; i++)
        encoded_dense_mask[i] = 1.0;

    {
        CAHELGPUPlaintext pt;
        encoder.encode(context, encoded_dense_mask, scale, pt);
        mod_switch_to_next_inplace(context, pt);
        mod_switch_to_next_inplace(context, pt);
        mod_switch_to_next_inplace(context, pt);
        mod_switch_to_next_inplace(context, pt);
        encoded_weights.dense_mask = pt;
    }
    cout << "dense_mask encoded" << endl;


    //------------------------------dense_b-----------------------------
    vec_t encoded_dense_b(8192, 0.0);
    for (size_t i = 0; i < 128; i++)
        encoded_dense_b[i] = weights.dense_b[i];

    {
        CAHELGPUPlaintext pt;
        encoder.encode(context, encoded_dense_b, scale, pt);
        mod_switch_to_next_inplace(context, pt);
        mod_switch_to_next_inplace(context, pt);
        mod_switch_to_next_inplace(context, pt);
        mod_switch_to_next_inplace(context, pt);
        mod_switch_to_next_inplace(context, pt);
        encoded_weights.dense_b = pt;
    }
    cout << "dense_b encoded" << endl;


    //------------------------------dense1_w-----------------------------
    vec_t encoded_dense1_w(8192, 0.0);
    for (size_t i = 0; i < 10 * 128; i++)
        encoded_dense1_w[i] = weights.dense1_w[i];

    {
        CAHELGPUPlaintext pt;
        encoder.encode(context, encoded_dense1_w, scale, pt);
        mod_switch_to_next_inplace(context, pt);
        mod_switch_to_next_inplace(context, pt);
        mod_switch_to_next_inplace(context, pt);
        mod_switch_to_next_inplace(context, pt);
        mod_switch_to_next_inplace(context, pt);
        mod_switch_to_next_inplace(context, pt);
        encoded_weights.dense1_w = pt;
    }
    cout << "dense1_w encoded" << endl;


    //------------------------------dense1_b-----------------------------
    vec_t encoded_dense1_b(8192, 0.0);
    for (size_t i = 0; i < 10; i++) {
        encoded_dense1_b[128 * i] = weights.dense1_b[i];
    }

    {
        CAHELGPUPlaintext pt;
        encoder.encode(context, encoded_dense1_b, scale, pt);
        mod_switch_to_next_inplace(context, pt);
        mod_switch_to_next_inplace(context, pt);
        mod_switch_to_next_inplace(context, pt);
        mod_switch_to_next_inplace(context, pt);
        mod_switch_to_next_inplace(context, pt);
        mod_switch_to_next_inplace(context, pt);
        mod_switch_to_next_inplace(context, pt);
        encoded_weights.dense1_b = pt;
    }
    cout << "dense1_b encoded" << endl;


    return encoded_weights;
}

auto encode_and_encrypt_image(const CAHELGPUContext &context, CAHELGPUCKKSEncoder &encoder,
                              const CAHELGPUSecretKey &secret_key, const vec_t &image) {
    vector<vec_t> kernel_wise_vec(9);
    double scale = std::pow(2.0, 40);

    for (size_t block_h = 0; block_h < 8; block_h++) {
        for (size_t block_w = 0; block_w < 8; block_w++) {
            for (size_t pool_h = 0; pool_h < 4; pool_h++) {
                for (size_t pool_w = 0; pool_w < 4; pool_w++) {
                    size_t h = block_h * 4 + pool_h;
                    size_t w = block_w * 4 + pool_w;
                    for (size_t kh = 0; kh < 3; kh++) {
                        for (size_t kw = 0; kw < 3; kw++) {
                            kernel_wise_vec[kh * 3 + kw].push_back(image[(h + kh) * 34 + (w + kw)]);
                        }
                    }
                }
            }
        }
    }

    // stride 1024 copy 7 times
    for (auto &k: kernel_wise_vec) {
        k.resize(8192);
        for (size_t i = 1; i < 8; i++) {
            for (size_t j = 0; j < 1024; j++) {
                k[1024 * i + j] = k[j];
            }
        }
    }

    // Encrypt image
    vector<CAHELGPUCiphertext> encrypted_image;
    for (const auto &k: kernel_wise_vec) {
        CAHELGPUPlaintext pt;
        encoder.encode(context, k, scale, pt);
        CAHELGPUCiphertext ct;
        secret_key.encrypt_symmetric(context, pt, ct, false);
        encrypted_image.push_back(ct);
    }

    return encrypted_image;
}

CAHELGPUCiphertext
infer_encrypted_image(const CAHELGPUContext &context, const CAHELGPURelinKey &rlk, const CAHELGPUGaloisKey &glk,
                      vector<CAHELGPUCiphertext> &encrypted_image, const struct_encoded_weights &encoded_weights) {

    assert(encrypted_image.size() == 9);
    double scale = std::pow(2.0, 40);

    CAHELGPUCiphertext ct;

    // -------------------------------------------conv--------------------------------------
    auto begin = std::chrono::steady_clock::now();

    vector<CAHELGPUCiphertext> vec_ct_conv(0);

    for (size_t i = 0; i < 9; i++) {
        CAHELGPUCiphertext ct_tmp;
        multiply_plain(context, encrypted_image[i], encoded_weights.conv_w[i], ct_tmp);
        rescale_to_next_inplace(context, ct_tmp);
        vec_ct_conv.push_back(ct_tmp);
    }

    add_many(context, vec_ct_conv, ct);
    ct.scale() = scale;

    add_plain_inplace(context, ct, encoded_weights.conv_b);

    auto end = std::chrono::steady_clock::now();
    auto t = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout << "conv: " << t << " us" << std::endl;

    begin = std::chrono::steady_clock::now();

    square_inplace(context, ct);
    relinearize_inplace(context, ct, rlk);
    rescale_to_next_inplace(context, ct);

    end = std::chrono::steady_clock::now();
    t = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout << "conv sq: " << t << " us" << std::endl;

    // --------------------------------------- avg pooling --------------------------------------
    begin = std::chrono::steady_clock::now();

    for (int i = 8; i > 0; i >>= 1) {
        CAHELGPUCiphertext ct_tmp;
        rotate_vector(context, ct, i, glk, ct_tmp);
        add_inplace(context, ct, ct_tmp);
    }

    multiply_plain_inplace(context, ct, encoded_weights.pool_mask);
    rescale_to_next_inplace(context, ct);

    end = std::chrono::steady_clock::now();
    t = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout << "avg pool: " << t << " us" << std::endl;

    // ---------------------------------------- dense ------------------------------------------
    begin = std::chrono::steady_clock::now();

    for (int i = 1; i < 16; i <<= 1) {
        CAHELGPUCiphertext ct_tmp;
        rotate_vector(context, ct, -i, glk, ct_tmp);
        add_inplace(context, ct, ct_tmp);
    }

    vector<CAHELGPUCiphertext> vec_ct_dense(0);
    for (size_t i = 0; i < 8; i++) {
        CAHELGPUCiphertext ct_tmp;
        multiply_plain(context, ct, encoded_weights.dense_w[i], ct_tmp);
        rescale_to_next_inplace(context, ct_tmp);

        for (int j = 4096; j >= 16; j >>= 1) {
            CAHELGPUCiphertext ct_tmp2;
            rotate_vector(context, ct_tmp, j, glk, ct_tmp2);
            add_inplace(context, ct_tmp, ct_tmp2);
        }

        multiply_plain_inplace(context, ct_tmp, encoded_weights.dense_mask);
        rescale_to_next_inplace(context, ct_tmp);
        vec_ct_dense.push_back(ct_tmp);
    }

    for (int i = 1; i < 8; i++)
        rotate_vector_inplace(context, vec_ct_dense[i], -16 * i, glk);
    add_many(context, vec_ct_dense, ct);

    end = std::chrono::steady_clock::now();
    t = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout << "dense: " << t << " us" << std::endl;

    begin = std::chrono::steady_clock::now();


    ct.scale() = scale;
    add_plain_inplace(context, ct, encoded_weights.dense_b);
    square_inplace(context, ct);
    relinearize_inplace(context, ct, rlk);
    rescale_to_next_inplace(context, ct);

    end = std::chrono::steady_clock::now();
    t = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout << "dense sq: " << t << " us" << std::endl;


    // ------------------------------------------- dense1 ------------------------------------------------------
    begin = std::chrono::steady_clock::now();

    CAHELGPUCiphertext ct_tmp_dense1 = ct;
    for (int i = 1; i < 10; i++) {
        CAHELGPUCiphertext ct_tmp;
        rotate_vector(context, ct_tmp_dense1, -128 * i, glk, ct_tmp);
        add_inplace(context, ct, ct_tmp);
    }

    multiply_plain_inplace(context, ct, encoded_weights.dense1_w);
    rescale_to_next_inplace(context, ct);
    for (int i = 64; i >= 1; i >>= 1) {
        CAHELGPUCiphertext ct_tmp;
        rotate_vector(context, ct, i, glk, ct_tmp);
        add_inplace(context, ct, ct_tmp);
    }
    ct.scale() = scale;
    add_plain_inplace(context, ct, encoded_weights.dense1_b);

    end = std::chrono::steady_clock::now();
    t = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout << "dense1: " << t << " us" << std::endl;

    return ct;
}

int main(int argc, char **argv) {
    string dataset;
    if (argc == 2) {
        dataset = argv[1];
    } else {
        cout << "Which dataset to test: ";
        cin >> dataset;
    }

    // -------------------------------SEAL setup------------------------------------
    cahel::EncryptionParameters parms(cahel::scheme_type::ckks);
    constexpr int log_n = 14;
    const size_t n = 1 << log_n;
    const std::vector<int> moduli_bits{60, 40, 40, 40, 40, 40, 40, 40, 60};
    auto moduli = cahel::CoeffModulus::Create(n, moduli_bits);
    parms.set_coeff_modulus(moduli);
    parms.set_poly_modulus_degree(n);

    const vector<int> galois_steps = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, -1, -2, -4, -8, -16,
                                      -32, -48, -64, -80, -96, -112, -128, -256, -384, -512, -640, -768, -896, -1024,
                                      -1152};
    auto galois_elts = get_elts_from_steps(galois_steps, n);
    parms.set_galois_elts(galois_elts);

    auto context = CAHELGPUContext(parms, true, cahel::sec_level_type::tc128);

    // Keygen
    CAHELGPUSecretKey sk(parms);
    sk.gen_secretkey(context);

    // Relin Keys
    CAHELGPURelinKey rlk(context);
    sk.gen_relinkey(context, rlk);

    // Galois Keys
    CAHELGPUGaloisKey glk(context);
    sk.create_galois_keys(context, glk);

    // encode
    CAHELGPUCKKSEncoder encoder(context);

    // -------------------------------load and encode weights------------------------------------
    string dataset_path = "datasets/" + dataset;
    string weights_file_path = "weights/" + dataset;

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

    auto encoded_weights = encode_weights(context, encoder, weights);


    // ------------------------------------ inference ------------------------------------------
    int matchedCount = 0;
    // long timeUsed = 0;

    // load dataset
    if (dataset == "mnist" || dataset == "fmnist") {
        auto [testImages, testLabels] = LoadMNISTDataset(dataset_path);
        for (size_t index = 0; index < testImages.size(); index++) {
            auto image = testImages[index];
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

            // encode and encrypt image
            auto encrypted_image = encode_and_encrypt_image(context, encoder, sk, zeropad_image);

            // timeUsed += t;
            auto result_encrypted_image = infer_encrypted_image(context, rlk, glk, encrypted_image, encoded_weights);

            CAHELGPUPlaintext result_image_pt;
            sk.decrypt(context, result_encrypted_image, result_image_pt);
            vec_t result_image;
            encoder.decode(context, result_image_pt, result_image);

            vec_t tmp(10);
            for (size_t c = 0; c < 10; c++)
                tmp[c] = result_image[128 * c];
            result_image = tmp;

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
            size_t height = 32;
            size_t width = 32;

            // 3in1
            vec_t new_image(1 * height * width);
            for (size_t h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    double sum = 0.0;
                    for (size_t c = 0; c < 3; c++) {
                        sum += uint8_image[c * height * width + h * width + w] / 255.0;
                    }
                    double avg = sum / 3.0;
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

            // encode and encrypt image
            auto encrypted_image = encode_and_encrypt_image(context, encoder, sk, zeropad_image);

            // timeUsed += t;
            auto result_encrypted_image = infer_encrypted_image(context, rlk, glk, encrypted_image, encoded_weights);

            CAHELGPUPlaintext result_image_pt;
            sk.decrypt(context, result_encrypted_image, result_image_pt);
            vec_t result_image;
            encoder.decode(context, result_image_pt, result_image);

            vec_t tmp(10);
            for (size_t c = 0; c < 10; c++)
                tmp[c] = result_image[128 * c];
            result_image = tmp;

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
}
