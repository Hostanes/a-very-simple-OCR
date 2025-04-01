
### Neural Network Architecture

3 Layer neural network + Input layer:
 - Input layer
 - Hidden layer 1
 - Hidden Layer 2
 - Output Layer

**Notation:** \[TODO]
- $a_{ij}$ is the activation of neuron $j$ on layer $i$
- $z_{ij}$ is the output of neuron $j$ of layer $i$
- $w_{ij}$ is the weight of the edge between neuron $i$ of 1 layer to neuron $j$ of the next layer during vector dot product operations

![[nn-arch.png]]

### Forward Propagation

1. Input image $x$ (784×1) → $z_1=W_1x+b_1z_1​=W_1​x+b_1$​ → ReLU → $a_1$​ (128×1).
2. $a_1 → z_2=W_2a_1+b_2z_2=W_2​a_1​+b_2$​ → ReLU → $a_2$​ (64×1).
3. $a_2​ → z_3=W_3a_2+b_3z_3​=W_3​a_2​+b_3​$ → Softmax → $a_3$ (10×1).    

The sizes of the Hidden layers $H1$ 128, and $H2$ 64, are arbitrary 

The **loss** (cross-entropy) compares $a3$ to the true label $y$.

### Backward Propagation

\[TODO]