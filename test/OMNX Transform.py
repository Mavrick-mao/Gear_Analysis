import torch
from torch import nn
import torch.onnx

from LSTM import Net


# PyTorchのモデルをロード
model = Net()
model.eval()

# 入力テンソルを作成
dummy_input = torch.randn(64, 1, 64, 44)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dummy_input = dummy_input.to(device)
model = model.to(device)

# ONNX形式でモデルをエクスポート
onnx_path = "cnn.onnx"
torch.onnx.export(model, dummy_input, onnx_path)

print(f"Model exported to {onnx_path}")