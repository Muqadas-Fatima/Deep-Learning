## 📅 Day 2: Foundations of Backpropagation & NumPy From Scratch

### 🎯 Objective
Demystify the "black box" of neural networks by moving away from high-level frameworks. Day 2 focused on mastering the mathematical mechanics of the chain rule, deriving backpropagation for a 2-layer architecture on paper, and writing a raw NumPy implementation to solve the non-linearly separable XOR problem.

---

### 🧠 Theoretical Core

#### 1. The Chain Rule
To understand how a network learns, we must calculate how changing an internal weight affects the total loss. If a change in weight affects an intermediate hidden state, which then affects the final prediction, the gradient is calculated by compounding these rates of change:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}$$

#### 2. Mathematical Derivation (2-Layer Network)
For a network with one hidden layer, the forward pass equations are defined as:

$$z_1 = XW_1 + b_1$$
$$a_1 = \sigma(z_1)$$
$$z_2 = a_1W_2 + b_2$$
$$\hat{y} = \sigma(z_2)$$

