# XOR Neural Network Learning Rate Experiment (PyTorch)

This project explores how **learning rate** affects the training performance of a simple neural network built in **PyTorch**.  
The model learns the **XOR logic gate**, and results are visualized through loss curves.  

---

## 🧠 Overview

This experiment demonstrates how different learning rates impact convergence and final loss in a small neural network trained on the XOR dataset.  
The script performs two tasks:
1. **Single Training Run** – trains once using a single learning rate (`run_and_plot_single_run()`).
2. **Learning Rate Comparison** – trains multiple times with varying learning rates and plots the results (`run_and_plot_comparison()`).

---

## 📁 Project Structure

| File | Description |
|------|--------------|
| `xor_experiment.py` | Main Python script containing the neural network and experiment logic. |
| `single_run_loss.png` | (Generated) Training loss vs. epochs for a single run. |
| `comparison_loss_plot.png` | (Generated) Comparison of training losses for multiple learning rates. |
| `requirements.txt` | Python dependencies for this project. |
| `README.md` | Documentation file (this file). |

---

## ⚙️ How It Works

### 1. Network Architecture
- **Input Layer:** 2 neurons (for XOR inputs)
- **Hidden Layer:** 4 neurons (configurable)
- **Output Layer:** 1 neuron (sigmoid activation)
- **Activation Function:** Sigmoid
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Stochastic Gradient Descent (SGD)

### 2. Dataset (XOR Table)

| Input | Output |
|--------|---------|
| [0, 0] | 0 |
| [0, 1] | 1 |
| [1, 0] | 1 |
| [1, 1] | 0 |

The network’s task is to predict the correct XOR output based on the two binary inputs.

---

####📊 Understanding the Results
Fast convergence: Large learning rates may cause quick initial drops in loss.

Instability: Too high a learning rate can make the model diverge.

Smooth, steady convergence: Moderate rates usually achieve balance.

Slow training: Very small learning rates lead to gradual but stable improvement.

Using a log-scale for loss helps visualize subtle differences when losses are very small.

####🔬 Example Analysis
At LR = 0.5, training may oscillate before stabilizing.

At LR = 0.1, loss decreases smoothly and steadily.

At LR = 0.01, training is stable but slower to converge.

This experiment highlights why tuning the learning rate is crucial in neural network optimization.

####💡 Possible Extensions
Save trained models with torch.save(model.state_dict(), "model_lr_0.1.pth").

Add a decision boundary visualization.

Compare different optimizers (e.g., Adam, RMSprop).

Plot accuracy in addition to loss.

Experiment with ReLU or Tanh activations.

####🧾 License
This project is released under the MIT License.
You are free to use, modify, and distribute it with attribution.

####🤝 Contributions
Contributions and suggestions are always welcome!
Open an issue or submit a pull request to discuss improvements.

