# Reducing the Transformer Architecture to a Minimum  

## Introduction  
This repository provides an implementation of the techniques introduced in the paper [**"Reducing the Transformer Architecture to a Minimum"**](https://arxiv.org/html/2410.13732v1). The paper simplifies the standard transformer model while preserving its strong performance.  

### Key Innovations:
1. **Removal of MLP layers:** Significantly reduces the number of trainable parameters. 
2. **Collapsing matrices:** Combines query-key and omiting value-projection matrices for streamlined architecture. ($ W_{qk}+noW_{v}W_{o} $ )
3. **Symmetric similarity matrices:** Enhances attention efficiency with fewer parameters. (symmetry)

These modifications achieve up to **90% reduction in parameters** while delivering competitive results on popular benchmarks, including MNIST, CIFAR-10, and ImageNet. This repository demonstrates how these techniques can be applied to build lightweight and efficient transformer models.  


---
## Different modifications 

Here we represent the modifications that we will be applying (apart from the removal of the MLP). 

The first image represents the traditional attention mechanism with the three matrices for queries, keys, and values, and with a final projection matrix.

The second image shows the collapsing of the query-key projection matrices ($W_{qk}$). This will reduce the number of parameters while keeping comparable performance with the original version.

The third figure represents the omission of the value-projection matrices. The justification for omitting $W_V$ and $W_O$ is based on the fact that in many NLP applications, we expect the output to be an embedding of a word or a language token. The space of embeddings is expected to be spanned by the input word embeddings. For that reason, it may seem unnecessary to transform the embeddings into another space and then transform them back to our embedding space. Therefore, we may choose to remove this transformation, and the output will be a convex combination of the input embeddings (which is expected to result in a valid and meaningful word).

Finally, the fourth image uses **Cholesky Decomposition**: Parameterize a lower triangular matrix $ T_{QK} $ and compute:

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

Here is a summary of results from 16 experiments on MNIST and CIFAR-10 using transformer models with varying configurations: 6 or 12 encoders, 1 or 4 attention heads, and with or without MLP.

<div style="text-align: center;">
    <img src="img\res_1.png" width="800" alt="Simplified Transformer Diagram: WQK">
</div>

The tables below represents loss and accuracy for different variants of transformer-encoder modifications on MNIST and CIFAR-10 respectively: 1 or 4 heads, with or without the MLP, with a single $W_{qk}$ matrix, no value and projection matrices, or a symmetric similarity measurement.

<div style="text-align: center;">
    <img src="img\res_2.png" width="800" alt="Simplified Transformer Diagram: WQK">
</div>

