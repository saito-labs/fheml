## Model Inference with FHE
Large transformer-based models like BERT, GPT, have achieved state-of-the-art (SOTA) performance across various real-world applications such as person re-identification, voice assistants , and code autocompletion. As these transformer models increasingly process sensitive data and tackle critical tasks, privacy issues have become a major concern during deployment. Private inference is designed to secure the model weights from users, while also ensuring that no user-specific private data is learned by the server.

## Building


CMake

### Building CAHEL


```
cmake -S. -Bbuild -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j8
sudo cmake --install build
```

### Building Xnet
```
cmake -S . -B build
cmake --build build -j8
```

## Examples

We introduce a compact network structure for real-time inference in convolutional neural networks based on FHE. We further propose several optimization strategies, including an innovative compression and encoding technique and rearrange ment in the pixel encoding sequence, enabling a highly efficient batched computation and reducing the demand for time-consuming HEoperations. To further expedite computation, we propose a GPU acceleration engine to leverage the massive thread-level parallelism to speed up computations.


```shell
./xnet-cahel
```





## Roadmap

🔨 = Pending

🛠 = Work In Progress

✅ = Feature complete


| Feature |  Status |
| ------- |  :------: |
| **Supported Model** |    |
| CNN for MNIST | ✅ |
| CNN for Cifar10 | ✅ |
| CNN for Healthcare System | ✅ |
| BERT-Tiny | 🛠 |
| LLaMA | 🛠|
| Traditional ML Algorithm (Decision Tree, KNN etc) | 🔨 |
| **Mode** |    |
| Inference| ✅ |
| Training | 🔨 |
| Fine-tuning | 🔨 |
| **Optimization** |    |
| GPU Acceleration | ✅ |
| **Functionality** |    |
| User-Friendly SDK| 🛠 |

## License

This code is MIT licensed.

Part of this code is borrowed from `ethereum-optimism/cannon`

Note: This code is unaudited. It in NO WAY should be used to secure any money until a lot more
testing and auditing are done. 
