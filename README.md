# ASH-net (Ahmouri Spectral Hybrid Network)

ASH-net is an innovative hybrid neural network framework utilizing a **Spectral Maintenance** protocol to ensure dynamic network stability during training. The project integrates a BiLSTM architecture for temporal sequence processing with Residual Blocks for spatial data, enhanced by a Self-Attention mechanism to boost contextual accuracy.

## 🚀 Key Features
* **Spectral Immunity:** Implementation of the `apply_spectral_maintenance` function to control variance floors and prevent gradient vanishing or explosion.
* **Hybrid Architecture:** A seamless integration of spatial and temporal processing stages.
* **Absolute Stability:** Designed with rigorous mathematical standards for weight matrix management.

## 🛠️ Requirements
For optimal performance, ensure you have the following library installed:

`pip install torch`

## 🏗️ Model Architecture
ASH-net relies on three fundamental stages:
1. **Spatial Stage:** Spatial feature processing via `ASResNetBlock`.
2. **Temporal Stage:** Extraction of temporal patterns using `AhmouriStabilizedBiLSTM`.
3. **Context Stage:** Context aggregation using `SimpleSelfAttention`.

## 📈 Quick Start
You can run the code directly to test the network's stability:

```python
from ash_net import ASHNet
import torch

# Initialize the model
model = ASHNet(feature_dim=64, num_classes=10)
model.apply_global_spectral_maintenance()

# Run a dummy forward pass
dummy_data = torch.randn(16, 128, 64)
output = model(dummy_data)
print(output.shape)
