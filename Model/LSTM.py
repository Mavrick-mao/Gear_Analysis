import torch
import torch.nn as nn

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.seq_len = 64  # 画像の Height を時系列のSequenceとしてLSTMに入力する
        self.feature_size = 44  # 画像の Width を特徴量の次元としてLSTMに入力する
        self.hidden_layer_size = 128  # 隠れ層のサイズ
        self.lstm_layers = 1  # LSTMのレイヤー数　(LSTMを何層重ねるか)

        self.lstm = nn.LSTM(self.feature_size,
                            self.hidden_layer_size,
                            num_layers=self.lstm_layers)

        self.fc = nn.Linear(self.hidden_layer_size, 10)

    def init_hidden_cell(self, batch_size, device):  # LSTMの隠れ層 hidden と記憶セル cell を初期化
        hedden = torch.zeros(self.lstm_layers, batch_size, self.hidden_layer_size).to(device)
        cell = torch.zeros(self.lstm_layers, batch_size, self.hidden_layer_size).to(device)
        return (hedden, cell)

    def forward(self, x):
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        batch_size = x.shape[0]

        self.hidden_cell = self.init_hidden_cell(batch_size, device)

        x = x.view(batch_size, self.seq_len,
                   self.feature_size)  # (Batch, Cannel, Height, Width) -> (Batch, Height, Width) = (Batch, Seqence, Feature)
        # 画像の Height を時系列のSequenceに、Width を特徴量の次元としてLSTMに入力する
        x = x.permute(1, 0, 2)  # (Batch, Seqence, Feature) -> (Seqence , Batch, Feature)

        lstm_out, (h_n, c_n) = self.lstm(x, self.hidden_cell)  # LSTMの入力データのShapeは(Seqence, Batch, Feature)
        # (h_n) のShapeは (num_layers, batch, hidden_size)
        x = h_n[-1, :, :]  # lstm_layersの最後のレイヤーを取り出す  (B, h)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    print(Net())