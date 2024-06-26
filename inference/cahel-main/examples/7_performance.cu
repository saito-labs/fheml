#include "example.h"
#include "util.cuh"

using namespace std;
using namespace cahel;

void bfv_bgv_performance_test(EncryptionParameters &parms) {
    CAHELGPUContext context(parms);
    print_parameters(context);
    cout << endl;
    auto cahel_context = context.cpu_context_;

    auto &context_data = (CAHELContext::ContextData &) (cahel_context->get_context_data(context.get_first_index()));
    auto &first_parms = (EncryptionParameters &) (context_data.parms());
    auto &plain_modulus = first_parms.plain_modulus();
    size_t poly_modulus_degree = first_parms.poly_modulus_degree();

    print_timer_banner();

    auto count = 100;

    CAHELGPUSecretKey secret_key(parms);
    {
        CUDATimer timer("gen_secretkey");
        for (auto i = 0; i < count; i++) {
            timer.start();
            secret_key.gen_secretkey(context);
            timer.stop();
        }
    }

    CAHELGPUPublicKey public_key(context);
    {
        CUDATimer timer("gen_publickey");
        for (auto i = 0; i < count; i++) {
            timer.start();
            secret_key.gen_publickey(context, public_key);
            timer.stop();
        }
    }

    CAHELGPURelinKey relin_keys(context);
    CAHELGPUGaloisKey gal_keys(context);

    // Generate relinearization keys
    {
        CUDATimer timer("gen_relinkey");
        for (auto i = 0; i < count; i++) {
            timer.start();
            secret_key.gen_relinkey(context, relin_keys);
            timer.stop();
        }
    }

    if (!cahel_context->get_context_data(0).qualifiers().using_batching) {
        cout << "Given encryption parameters do not support batching." << endl;
        return;
    }

    /*
    Generate Galois keys. In larger examples the Galois keys can use a lot of
    memory, which can be a problem in constrained systems. The user should
    try some larger runs of the test and observe their effect on the
    memory pool allocation size. The key generation can also take a long time,
    as can be observed from the print-out.
    */
    secret_key.create_galois_keys(context, gal_keys);

    CAHELGPUBatchEncoder batch_encoder(context);
    size_t slot_count = batch_encoder.slot_count();
    random_device rd;

    CAHELGPUPlaintext plain(context);

    /*
    Populate a vector of values to batch.
    */
    vector<int64_t> pod_vector;
    for (size_t i = 0; i < slot_count; i++) {
        pod_vector.push_back(static_cast<int64_t>(plain_modulus.reduce(rd())));
    }

    /*
    [Batching]
    There is nothing unusual here. We batch our random plaintext matrix
    into the polynomial. Note how the plaintext we create is of the exactly
    right size so unnecessary reallocations are avoided.
    */
    {
        CUDATimer timer("encode");
        for (auto i = 0; i < count; i++) {
            timer.start();
            batch_encoder.encode(context, pod_vector, plain);
            timer.stop();
        }
    }

    /*
    [Unbatching]
    We unbatch what we just batched.
    */
    vector<int64_t> pod_vector2(slot_count);
    {
        CUDATimer timer("decode");
        for (auto i = 0; i < count; i++) {
            timer.start();
            batch_encoder.decode(context, plain, pod_vector2);
            timer.stop();
        }
    }

    if (pod_vector2 != pod_vector) {
        throw runtime_error("Batch/unbatch failed. Something is wrong.");
    }

    /*
    [Encryption]
    We make sure our ciphertext is already allocated and large enough
    to hold the encryption with these encryption parameters. We encrypt
    our random batched matrix here.
    */
    CAHELGPUCiphertext encrypted(context);
    {
        CUDATimer timer("encrypt_asymmetric");
        for (auto i = 0; i < count; i++) {
            timer.start();
            public_key.encrypt_asymmetric(context, plain, encrypted, false);
            timer.stop();
        }
    }

    /*
    [Decryption]
    We decrypt what we just encrypted.
    */
    {
        CUDATimer timer("decrypt");
        for (auto i = 0; i < count; i++) {
            timer.start();
            secret_key.decrypt(context, encrypted, plain);
            timer.stop();
        }
    }

    // homomorphic operations

    CAHELGPUPlaintext plain1(context);
    CAHELGPUPlaintext plain2(context);
    CAHELGPUCiphertext encrypted1(context);
    batch_encoder.encode(context, vector<int64_t>(slot_count, 1), plain1);
    public_key.encrypt_asymmetric(context, plain1, encrypted1, false);
    CAHELGPUCiphertext encrypted2(context);
    batch_encoder.encode(context, vector<int64_t>(slot_count, 1), plain2);
    public_key.encrypt_asymmetric(context, plain2, encrypted2, false);

    /*
    [Add]
    We create two ciphertexts and perform a few additions with them.
    */
    {
        CUDATimer timer("add_inplace");
        for (auto i = 0; i < count; i++) {
            CAHELGPUCiphertext tmp_ct(encrypted1);
            timer.start();
            add_inplace(context, tmp_ct, encrypted2);
            timer.stop();
        }
    }

    /*
    [Add Plain]
    */
    {
        CUDATimer timer("add_plain");
        for (auto i = 0; i < count; i++) {
            CAHELGPUCiphertext tmp_ct(encrypted1);
            timer.start();
            add_plain_inplace(context, tmp_ct, plain);
            timer.stop();
        }
    }

    /*
    [Multiply]
    We multiply two ciphertexts. Since the size of the result will be 3,
    and will overwrite the first argument, we reserve first enough memory
    to avoid reallocating during multiplication.
    */
    {
        CUDATimer timer("multiply");
        for (auto i = 0; i < count; i++) {
            CAHELGPUCiphertext tmp_ct(encrypted1);
            timer.start();
            multiply_inplace(context, tmp_ct, encrypted2);
            relinearize_inplace(context, tmp_ct, relin_keys);
            timer.stop();
        }
    }

    /*
    [Multiply Plain]
    We multiply a ciphertext with a random plaintext. Recall that
    multiply_plain does not change the size of the ciphertext, so we use
    encrypted2 here.
    */
    {
        CUDATimer timer("multiply_plain");
        for (auto i = 0; i < count; i++) {
            CAHELGPUCiphertext tmp_ct(encrypted1);
            timer.start();
            multiply_plain_inplace(context, tmp_ct, plain);
            timer.stop();
        }
    }

    /*
    [Rotate Rows One Step]
    We rotate matrix rows by one step left and measure the time.
    */
    {
        CUDATimer timer("rotate_rows_inplace_one_step");
        for (auto i = 0; i < count; i++) {
            CAHELGPUCiphertext tmp_ct(encrypted1);
            timer.start();
            rotate_rows_inplace(context, tmp_ct, 1, gal_keys);
            timer.stop();
        }
    }
}

void bfv_bgv_performance(scheme_type scheme, mul_tech_type mul_tech) {
    print_example_banner("BFV Performance Test with Degrees: 4096, 8192, 16384, 32768, and 65536");

    if (scheme != scheme_type::bfv && scheme != scheme_type::bgv) {
        throw std::invalid_argument("This function is only for BFV and BGV.");
    }

    if (scheme == scheme_type::bfv && mul_tech == mul_tech_type::none) {
        throw std::invalid_argument("For BFV benchmarking, mul tech must be set.");
    }

    // 2 ^ 12

    {
        EncryptionParameters parms(scheme);
        size_t poly_modulus_degree = (1 << 12);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(
                poly_modulus_degree, {
                        35, 35,

                        38}));
        parms.set_special_modulus_size(1);
        if (scheme == scheme_type::bfv) parms.set_mul_tech(mul_tech);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        bfv_bgv_performance_test(parms);
    }

    // 2 ^ 13

    {
        cout << endl;
        EncryptionParameters parms(scheme);
        size_t poly_modulus_degree = (1 << 13);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(
                poly_modulus_degree, {
                        54, 54, 54,

                        56}));
        parms.set_special_modulus_size(1);
        if (scheme == scheme_type::bfv) parms.set_mul_tech(mul_tech);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        bfv_bgv_performance_test(parms);
    }

    {
        cout << endl;
        EncryptionParameters parms(scheme);
        size_t poly_modulus_degree = (1 << 13);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(
                poly_modulus_degree, {
                        36, 36, 36, 36, 36,

                        38}));
        parms.set_special_modulus_size(1);
        if (scheme == scheme_type::bfv) parms.set_mul_tech(mul_tech);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        bfv_bgv_performance_test(parms);
    }

    {
        cout << endl;
        EncryptionParameters parms(scheme);
        size_t poly_modulus_degree = (1 << 13);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(
                poly_modulus_degree, {
                        36, 36, 36, 36,

                        37, 37}));
        parms.set_special_modulus_size(2);
        if (scheme == scheme_type::bfv) parms.set_mul_tech(mul_tech);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        bfv_bgv_performance_test(parms);
    }

    // 2 ^ 14

    {
        cout << endl;
        EncryptionParameters parms(scheme);
        size_t poly_modulus_degree = (1 << 14);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(
                poly_modulus_degree, {
                        54, 54, 54, 54, 54, 54, 54,

                        60}));
        parms.set_special_modulus_size(1);
        if (scheme == scheme_type::bfv) parms.set_mul_tech(mul_tech);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        bfv_bgv_performance_test(parms);
    }

    {
        cout << endl;
        EncryptionParameters parms(scheme);
        size_t poly_modulus_degree = (1 << 14);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(
                poly_modulus_degree, {
                        36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36,

                        42}));
        parms.set_special_modulus_size(1);
        if (scheme == scheme_type::bfv) parms.set_mul_tech(mul_tech);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        bfv_bgv_performance_test(parms);
    }

    {
        cout << endl;
        EncryptionParameters parms(scheme);
        size_t poly_modulus_degree = (1 << 14);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(
                poly_modulus_degree, {
                        36, 36, 36, 36, 36, 36, 36, 36,

                        37, 37, 38, 38}));
        parms.set_special_modulus_size(4);
        if (scheme == scheme_type::bfv) parms.set_mul_tech(mul_tech);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        bfv_bgv_performance_test(parms);
    }

    // 2 ^ 15

    {
        cout << endl;
        EncryptionParameters parms(scheme);
        size_t poly_modulus_degree = (1 << 15);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(
                poly_modulus_degree, {
                        55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55,

                        56}));
        parms.set_special_modulus_size(1);
        if (scheme == scheme_type::bfv) parms.set_mul_tech(mul_tech);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        bfv_bgv_performance_test(parms);
    }

    {
        cout << endl;
        EncryptionParameters parms(scheme);
        size_t poly_modulus_degree = (1 << 15);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(
                poly_modulus_degree, {
                        36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36,

                        53}));
        parms.set_special_modulus_size(1);
        if (scheme == scheme_type::bfv) parms.set_mul_tech(mul_tech);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        bfv_bgv_performance_test(parms);
    }

    {
        cout << endl;
        EncryptionParameters parms(scheme);
        size_t poly_modulus_degree = (1 << 15);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(
                poly_modulus_degree, {
                        36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36,

                        38, 38, 38, 38, 38, 38, 38, 39}));
        parms.set_special_modulus_size(8);
        if (scheme == scheme_type::bfv) parms.set_mul_tech(mul_tech);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        bfv_bgv_performance_test(parms);
    }

    // 2 ^ 16

    {
        cout << endl;
        EncryptionParameters parms(scheme);
        size_t poly_modulus_degree = (1 << 16);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(
                poly_modulus_degree, {
                        55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55,
                        56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56,

                        56}));
        parms.set_special_modulus_size(1);
        if (scheme == scheme_type::bfv) parms.set_mul_tech(mul_tech);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        bfv_bgv_performance_test(parms);
    }

    {
        cout << endl;
        EncryptionParameters parms(scheme);
        size_t poly_modulus_degree = (1 << 16);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(
                poly_modulus_degree, {
                        36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
                        37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
                        37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
                        37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37,

                        48}));
        parms.set_special_modulus_size(1);
        if (scheme == scheme_type::bfv) parms.set_mul_tech(mul_tech);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        bfv_bgv_performance_test(parms);
    }

    {
        cout << endl;
        EncryptionParameters parms(scheme);
        size_t poly_modulus_degree = (1 << 16);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(
                poly_modulus_degree, {
                        36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
                        36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
                        36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36,

                        39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 40}));
        parms.set_special_modulus_size(16);
        if (scheme == scheme_type::bfv) parms.set_mul_tech(mul_tech);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        bfv_bgv_performance_test(parms);
    }

    /*

    // 2 ^ 17

    {
        cout << endl;
        EncryptionParameters parms(scheme);
        size_t poly_modulus_degree = (1 << 17);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(
                poly_modulus_degree, {
                        55, 55, 55, 55, 55, 55, 55, 55, 55, 55,
                        56, 56, 56, 56, 56, 56, 56, 56, 56, 56,
                        56, 56, 56, 56, 56, 56, 56, 56, 56, 56,
                        56, 56, 56, 56, 56, 56, 56, 56, 56, 56,
                        56, 56, 56, 56, 56, 56, 56, 56, 56, 56,
                        56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56,

                        58}));
        parms.set_special_modulus_size(1);
        if (scheme == scheme_type::bfv) parms.set_mul_tech(mul_tech);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        bfv_bgv_performance_test(parms);
    }

    {
        cout << endl;
        EncryptionParameters parms(scheme);
        size_t poly_modulus_degree = (1 << 17);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(
                poly_modulus_degree, {
                        37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
                        37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
                        37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
                        37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
                        37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
                        37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
                        37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
                        38, 38, 38, 38, 38, 38, 38, 38, 38, 38,
                        38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38,

                        40}));
        parms.set_special_modulus_size(1);
        if (scheme == scheme_type::bfv) parms.set_mul_tech(mul_tech);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        bfv_bgv_performance_test(parms);
    }

    {
        cout << endl;
        EncryptionParameters parms(scheme);
        size_t poly_modulus_degree = (1 << 17);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(
                poly_modulus_degree, {
                        37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
                        37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
                        37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
                        37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
                        37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
                        37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37,

                        37, 37, 37, 37, 37, 37, 37, 37,
                        38, 38, 38, 38, 38, 38, 38, 38, 38, 38,
                        38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38}));
        parms.set_special_modulus_size(32);
        if (scheme == scheme_type::bfv) parms.set_mul_tech(mul_tech);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        bfv_bgv_performance_test(parms);
    }
     */

}

void ckks_performance_test(EncryptionParameters &parms, double scale) {
    CAHELGPUContext context(parms);
    print_parameters(context);
    cout << endl;
    auto cahel_context = context.cpu_context_;

    auto &context_data = (CAHELContext::ContextData &) (cahel_context->get_context_data(context.get_first_index()));
    auto &first_parms = (EncryptionParameters &) (context_data.parms());
    size_t poly_modulus_degree = first_parms.poly_modulus_degree();

    print_timer_banner();

    auto count = 100;

    CAHELGPUSecretKey secret_key(parms);
    {
        CUDATimer timer("gen_secretkey");
        for (auto i = 0; i < count; i++) {
            timer.start();
            secret_key.gen_secretkey(context);
            timer.stop();
        }
    }

    CAHELGPUPublicKey public_key(context);
    {
        CUDATimer timer("gen_publickey");
        for (auto i = 0; i < count; i++) {
            timer.start();
            secret_key.gen_publickey(context, public_key);
            timer.stop();
        }
    }

    CAHELGPURelinKey relin_keys(context);
    CAHELGPUGaloisKey gal_keys(context);
    // Generate relinearization keys
    {
        CUDATimer timer("gen_relinkey");
        for (auto i = 0; i < count; i++) {
            timer.start();
            secret_key.gen_relinkey(context, relin_keys);
            timer.stop();
        }
    }

    if (!cahel_context->get_context_data(0).qualifiers().using_batching) {
        cout << "Given encryption parameters do not support batching." << endl;
        return;
    }

    secret_key.create_galois_keys(context, gal_keys);

    CAHELGPUCKKSEncoder ckks_encoder(context);

    /*
    Populate a vector of floating-point values to batch.
    */
    std::vector<cuDoubleComplex> pod_vector;
    random_device rd;
    pod_vector.reserve(ckks_encoder.slot_count());
    for (size_t i = 0; i < ckks_encoder.slot_count(); i++)
        pod_vector.push_back(make_cuDoubleComplex(double(1.01), double(1.01)));
    std::vector<cuDoubleComplex> pod_vector2;
    std::vector<cuDoubleComplex> pod_vector3;
    pod_vector3.resize(ckks_encoder.slot_count());
    std::vector<cuDoubleComplex> pod_vector4;
    pod_vector4.resize(ckks_encoder.slot_count());

    /*
    [Encoding]
    For scale we use the square root of the last coeff_modulus prime
    from parms.
    */
    CAHELGPUPlaintext plain(context);

    {
        CUDATimer timer("encode");
        for (auto i = 0; i < count; i++) {
            timer.start();
            ckks_encoder.encode(context, pod_vector, scale, plain);
            timer.stop();
        }
    }

    /*
    [Decoding]
    */
    {
        CUDATimer timer("decode");
        for (auto i = 0; i < count; i++) {
            timer.start();
            ckks_encoder.decode(context, plain, pod_vector2);
            timer.stop();
        }
    }

    /*
    [Encryption]
    */
    CAHELGPUCiphertext encrypted(context);
    {
        CUDATimer timer("encrypt_asymmetric");
        for (auto i = 0; i < count; i++) {
            timer.start();
            public_key.encrypt_asymmetric(context, plain, encrypted, false);
            timer.stop();
        }
    }

    /*
    [Decryption]
    */
    CAHELGPUPlaintext plain2(context);
    {
        CUDATimer timer("decrypt");
        for (auto i = 0; i < count; i++) {
            timer.start();
            secret_key.decrypt(context, encrypted, plain2);
            timer.stop();
        }
    }

    // homomorphic operations

    CAHELGPUCiphertext encrypted1(context);
    for (size_t j = 0; j < ckks_encoder.slot_count(); j++)
        pod_vector3[j] = make_cuDoubleComplex(double(1), double(0));
    ckks_encoder.encode(context, pod_vector3, scale, plain);
    public_key.encrypt_asymmetric(context, plain, encrypted1, false);

    CAHELGPUCiphertext encrypted2(context);
    for (size_t j = 0; j < ckks_encoder.slot_count(); j++)
        pod_vector4[j] = make_cuDoubleComplex(double(1), double(0));
    ckks_encoder.encode(context, pod_vector4, scale, plain2);
    public_key.encrypt_asymmetric(context, plain2, encrypted2, false);

    /*
    [Add]
    */
    {
        CUDATimer timer("add");
        for (auto i = 0; i < count; i++) {
            CAHELGPUCiphertext tmp_ct(encrypted1);
            timer.start();
            add_inplace(context, tmp_ct, encrypted2);
            timer.stop();
        }
    }

    /*
    [Add Plain]
    */
    {
        CUDATimer timer("add_plain");
        for (auto i = 0; i < count; i++) {
            CAHELGPUCiphertext tmp_ct(encrypted1);
            timer.start();
            add_plain_inplace(context, tmp_ct, plain);
            timer.stop();
        }
    }

    /*
    [Multiply]
    */
    {
        CUDATimer timer("multiply");
        for (auto i = 0; i < count; i++) {
            CAHELGPUCiphertext tmp_ct(encrypted1);
            timer.start();
            multiply_inplace(context, tmp_ct, encrypted2);
            relinearize_inplace(context, tmp_ct, relin_keys);
            timer.stop();
        }
    }

    /*
    [Multiply Plain]
    */
    {
        CUDATimer timer("multiply_plain");
        for (auto i = 0; i < count; i++) {
            CAHELGPUCiphertext tmp_ct(encrypted1);
            timer.start();
            multiply_plain_inplace(context, tmp_ct, plain);
            timer.stop();
        }
    }

    /*
    [Rescale]
    */
    {
        CUDATimer timer("rescale_to_next");
        for (auto i = 0; i < count; i++) {
            CAHELGPUCiphertext tmp_ct(encrypted1);
            multiply_inplace(context, tmp_ct, encrypted2);
            relinearize_inplace(context, tmp_ct, relin_keys);
            timer.start();
            rescale_to_next_inplace(context, tmp_ct);
            timer.stop();
        }
    }

    /*
    [Rotate Vector]
    */
    {
        CUDATimer timer("rotate_vector_one_step");
        for (auto i = 0; i < count; i++) {
            CAHELGPUCiphertext tmp_ct(encrypted1);
            timer.start();
            rotate_vector_inplace(context, tmp_ct, 1, gal_keys);
            timer.stop();
        }
    }
}

void ckks_performance() {
    print_example_banner("CKKS Performance Test with Degrees: 4096, 8192, 16384, 32768, and 65536");

    // 2 ^ 13

    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 1 << 13;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {
                60, 40,
                60
        }));
        parms.set_special_modulus_size(1);
        double scale = pow(2.0, 40);
        ckks_performance_test(parms, scale);
    }

    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 1 << 13;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {
                60, 40, 40,
                60
        }));
        parms.set_special_modulus_size(1);
        double scale = pow(2.0, 40);
        ckks_performance_test(parms, scale);
    }

    // 2 ^ 14

    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 1 << 14;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {
                60, 40, 40, 40, 40, 40, 40, 40,
                60
        }));
        parms.set_special_modulus_size(1);
        double scale = pow(2.0, 40);
        ckks_performance_test(parms, scale);
    }

    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 1 << 14;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {
                60, 40, 40, 40, 40, 40,
                60, 60
        }));
        parms.set_special_modulus_size(2);
        double scale = pow(2.0, 40);
        ckks_performance_test(parms, scale);
    }

    // 2 ^ 15

    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 1 << 15;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {
                60, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                60
        }));
        parms.set_special_modulus_size(1);
        double scale = pow(2.0, 40);
        ckks_performance_test(parms, scale);
    }

    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 1 << 15;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {
                60, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                40, 40, 40, 40, 40, 40, 40, 40,
                60, 60
        }));
        parms.set_special_modulus_size(2);
        double scale = pow(2.0, 40);
        ckks_performance_test(parms, scale);
    }

    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 1 << 15;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {
                60, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                40, 40, 40, 40, 40,
                60, 60, 60
        }));
        parms.set_special_modulus_size(3);
        double scale = pow(2.0, 40);
        ckks_performance_test(parms, scale);
    }

    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 1 << 15;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {
                60, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                40, 40,
                60, 60, 60, 60
        }));
        parms.set_special_modulus_size(4);
        double scale = pow(2.0, 40);
        ckks_performance_test(parms, scale);
    }

    // 2^ 16

    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 1 << 16;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {
                60, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                40, 40,
                60
        }));
        parms.set_special_modulus_size(1);
        double scale = pow(2.0, 40);
        ckks_performance_test(parms, scale);
    }

    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 1 << 16;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {
                60, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                60, 60
        }));
        parms.set_special_modulus_size(2);
        double scale = pow(2.0, 40);
        ckks_performance_test(parms, scale);
    }

    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 1 << 16;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {
                60, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                40, 40, 40, 40, 40, 40, 40, 40, 40,
                60, 60, 60
        }));
        parms.set_special_modulus_size(3);
        double scale = pow(2.0, 40);
        ckks_performance_test(parms, scale);
    }

    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 1 << 16;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {
                60, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                40, 40, 40, 40, 40, 40,
                60, 60, 60, 60
        }));
        parms.set_special_modulus_size(4);
        double scale = pow(2.0, 40);
        ckks_performance_test(parms, scale);
    }

    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 1 << 16;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {
                60, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                40, 40, 40, 40, 40,
                60, 60, 60, 60, 60
        }));
        parms.set_special_modulus_size(5);
        double scale = pow(2.0, 40);
        ckks_performance_test(parms, scale);
    }

    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 1 << 16;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {
                60, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                60, 60, 60, 60, 60, 60
        }));
        parms.set_special_modulus_size(6);
        double scale = pow(2.0, 40);
        ckks_performance_test(parms, scale);
    }
}

/*
Prints a sub-menu to select the performance test.
*/
void example_performance_test() {
    print_example_banner("Example: Performance Test");

    while (true) {
        cout << endl;
        cout << "Select a scheme:" << endl;
        cout << "  1. BFV BEHZ" << endl;
        cout << "  2. BFV HPS" << endl;
        cout << "  3. BGV" << endl;
        cout << "  4. CKKS" << endl;
        cout << "  0. Back to main menu" << endl;

        int selection = 0;
        cout << endl
             << "> Run performance test (1 ~ 4) or go back (0): ";
        if (!(cin >> selection)) {
            cout << "Invalid option." << endl;
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            continue;
        }

        switch (selection) {
            case 1:
                bfv_bgv_performance(cahel::scheme_type::bfv, cahel::mul_tech_type::behz);
                break;

            case 2:
                bfv_bgv_performance(cahel::scheme_type::bfv, cahel::mul_tech_type::hps);
                break;

            case 3:
                bfv_bgv_performance(cahel::scheme_type::bgv);
                break;

            case 4:
                ckks_performance();
                break;

            case 0:
                cout << endl;
                return;

            default:
                cout << "Invalid option." << endl;
        }
    }
}
