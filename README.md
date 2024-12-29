#  Reducing the Transformer Architecture to a Minimum

## Introduction  
This repository provides an implementation of the techniques introduced in the paper [**"Reducing the Transformer Architecture to a Minimum"**](https://arxiv.org/html/2410.13732v1). The paper simplifies the standard transformer model while preserving its strong performance.  

### Key Innovations:
1. **Removal of MLP layers:** Significantly reduces the number of trainable parameters. 
2. **Collapsing matrices:** Combines query-key and omiting value-projection matrices for streamlined architecture. ($W_{qk}+noW_{v}W_{o}$ )
3. **Symmetric similarity matrices:** Enhances attention efficiency with fewer parameters. (symmetry)

These modifications achieve up to **90% reduction in parameters** while delivering competitive results on popular benchmarks, including MNIST, CIFAR-10, and ImageNet. This repository demonstrates how these techniques can be applied to build lightweight and efficient transformer models.  


---
## Different modifications 

Here we represent the modifications that we will be applying (apart from the removal of the MLP). 

The first image represents the traditional attention mechanism with the three matrices for queries, keys, and values, and with a final projection matrix.

The second image shows the collapsing of the query-key projection matrices ($W_{qk}$). This will reduce the number of parameters while keeping comparable performance with the original version.

The third figure represents the omission of the value-projection matrices. The justification for omitting $W_V$ and $W_O$ is based on the fact that in many NLP applications, we expect the output to be an embedding of a word or a language token. The space of embeddings is expected to be spanned by the input word embeddings. For that reason, it may seem unnecessary to transform the embeddings into another space and then transform them back to our embedding space. Therefore, we may choose to remove this transformation, and the output will be a convex combination of the input embeddings (which is expected to result in a valid and meaningful word).

Finally, the fourth image uses **Cholesky Decomposition**: Parameterize a lower triangular matrix $T_{QK}$ and compute:

$$
W_{QK_s} = T_{QK_s} (T_{QK_s})^T
$$


This ensures the symmetry of the similarity matrix, which will allow us to learn only half of the matrix.


<img src="img\unchanged.png" width="600" alt="Simplified Transformer Diagram: Unchanged">

Figure 1: Traditional Attention Mechanism.

<img src="img\wqk.png" width="600" alt="Simplified Transformer Diagram: WQK">

Figure 2: Query and key matrices are collapsed into a single matrix of the same size.

<img src="img\omission.png" width="600" alt="Simplified Transformer Diagram: Omission">

Figure 3: In addition to the collapsed query and key matrices, value and projection matrices, are omitted without eliminating the substance of the attention mechanism

<img src="img\symmetry.png" width="600" alt="Simplified Transformer Diagram: Symmetry">

Figure 4: The symmetric definition of a similarity matrix requires only half the parameters. This can be achieved by parameterizing a lower triangular matrix and multiplying it by its transpose

---

## Usage  
To get started, follow these steps:  
1. **Clone the repository** and install dependencies.  
2. Modify the `config.py` file to:
   - Define the dataset for benchmarking.
   - Specify combinations of different transformer architectures for experiments.  

3. **Run experiments** and log results to Weights & Biases (WandB) using the following command:  
   ```bash
   python main.py
## Results

As the model complexity increases (i.e., the number of parameters increases), we need more training data to ensure that our model reaches its full potential. Otherwise, the model is likely to exhibit high variance due to an insufficient training dataset relative to the number of parameters, leading to overfitting. Therefore, it is crucial to ensure that the size of our training data aligns with the complexity of the model, enabling us to accurately assess its performance.
This aligns with the requirement for the overdetermination ratio $Q$ to be greater than unity, which is defined as: $Q=\dfrac{KM}{P}$
where $K$ is the number of training examples, $M$ is the length of the output vector (usually equal to the number of classes), and $P$ is the number of trainable model parameters.

Here is a summary of results from 16 experiments on MNIST and CIFAR-10 using transformer models with varying configurations: 6 or 12 encoders, 1 or 4 attention heads, and with or without MLP.

<div style="text-align: center;">
    <img src="img\res_1.png" width="800" alt="Simplified Transformer Diagram: WQK">
</div>

**Comments** :
- On MNIST, removing the MLP is a highly efficient trade-off, with minimal performance loss (<0.5%) and significant parameter reductions (~33%).
- On CIFAR-10, the impact of removing the MLP is more pronounced, with performance losses ranging from ~2% to 9%, depending on the configuration. However, configurations like 12-4 and 6-4 show a relatively favorable balance.
- The performance with 12 encoders is not superior to that with 6 encoders, for that reason, only 6 encoders will be used for the following experiments.


The tables below represent loss and accuracy for different variants of transformer-encoder modifications on MNIST and CIFAR-10 respectively: 1 or 4 heads, with or without the MLP, with a single $W_{qk}$ matrix, no value and projection matrices, or a symmetric similarity measurement.

<div style="text-align: center;">
    <img src="img\res_2.png" width="800" alt="Simplified Transformer Diagram: WQK">
</div>


### MNIST:
- Using a collapsed matrix $W_{qk}$, while keeping the MLP intact, results in better validation accuracy (**0.061% improvement**) with a significant parameter reduction of approximately **16%**. Adding the omission of $W_v$ and $W_O$ further reduces the parameters by nearly **48%**, with only a small performance loss of **0.11%**.

- Removing only the MLP while maintaining the rest of the architecture reduces the number of parameters by approximately **32%**, with a relatively small performance loss of **0.79%**.

- For the symmetric variant, the configuration with 1 head shows subpar performance (potentially requiring more epochs to converge). However, the 4-head configuration achieves a validation accuracy of **97.49%**, comparable to the asymmetric similarity variant without the MLP (**97.69%**). Notably, the symmetric 4-head configuration achieves a **62% parameter reduction**, while the asymmetric variant achieves **48%**.

- The final variant, without the MLP and with both collapsed and omitted matrices, achieves a validation accuracy of **94.45%** (a performance loss of **3.94%**) while delivering an **80% reduction in parameters**, making it the most parameter-efficient configuration.


### CIFAR-10:
- Using a collapsed matrix $W_{qk}$, while keeping the MLP intact, along with omitted matrices $W_v$ and $W_O$, results in a parameter reduction of approximately **45\%**, with a relatively small performance loss of **4.47\%**.

- Removing only the MLP while maintaining the rest of the architecture reduces the number of parameters by approximately **30\%**, with a performance loss of **10\%** (from 68.19\% validation accuracy to 61.37\% for the 4-head version).

- The validation accuracy achieved by the symmetric variant for 4 heads is 46.84\%, which corresponds to a performance loss of **31\%**, a relatively large loss even though the parameter reduction is **58\%**. This may be due to the fact that this architecture is undertrained and requires more epochs for training (as we trained it for only 100 epochs, compared to 500 epochs in the original paper, due to limited compute power), but also to the fact that CIFAR-10 is not as easy a benchmark as MNIST.


## Conclusion

The results demonstrate that collapsing and omitting matrices, combined with the removal of MLPs, can significantly reduce the number of parameters without significantly affecting performance. Depending on the use case and its difficulty, one can balance parameter efficiency and performance by using different variants.

