from torch import nn
from torch.utils.data import DataLoader
import torch
import os
import wandb
from tqdm import tqdm
import datetime

from Mel_dataset import AudioDataset
from Model.NNmodel.CNN_2 import Model

#all changeable parameters
#------------------------------------------------------------
#for wave loading
Train_path = r"C:\Users\maver\PycharmProjects\pythonProject1\Sampling Data\train_data.csv"
Test_path = r"C:\Users\maver\PycharmProjects\pythonProject1\Sampling Data\test_data.csv"

SAMPLE_RATE = 22050
NUM_SAMPLES = 22050

#for ML
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001

#for wandb plot
dt_now = datetime.datetime.now()
os.environ["WANDB_API_KEY"] = 'babea74b5fe94390da26fe62b1bfb073f3d72215'
PROJECT = "Audio"
GROUP = "8class/CNN"
NAME = f"{dt_now.strftime('%Y年%m月%d日 %H:%M:%S')}"
#-----------------------------------------------------------

def train_model(net, dataloaders_dict, criterion, optimizer, device, num_epochs):
    # epochのループ
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-------------')

        single_train_epoch_loss = 0
        single_train_epoch_acc = 0

        single_val_epoch_loss = 0
        single_val_epoch_acc = 0

        # epochごとの学習と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
            else:
                net.eval()  # モデルを検証モードに

            epoch_loss = 0.0  # epochの損失和
            epoch_corrects = 0  # epochの正解数

            # 未学習時の検証性能を確かめるため、epoch=0の訓練は省略
            if (epoch == 0) and (phase == 'train'):
                continue

            # データローダーからミニバッチを取り出すループ
            for i, (inputs, labels) in tqdm(enumerate(dataloaders_dict[phase])):

                # GPUを使用する場合は明示的に指定
                # print(inputs.shape)
                inputs = inputs.to(device)
                labels = labels.to(device)
                # optimizerを初期化
                optimizer.zero_grad()

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):  # 訓練モードのみ勾配を算出
                    outputs = net(inputs)  # 順伝播
                    # print(outputs.shape)
                    loss = criterion(outputs, labels)  # 損失を計算
                    _, preds = torch.max(outputs, 1)  # ラベルを予測

                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # イタレーション結果の計算
                    # lossの合計を更新
                    epoch_loss += loss.item() * inputs.size(0)
                    # 正解数の合計を更新
                    epoch_corrects += torch.sum(preds == labels.data)

            # epochごとのlossと正解率を表示
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = torch.tensor(epoch_corrects, dtype=torch.float64).clone().detach() / len(dataloaders_dict[phase].dataset)

            wandb.log({phase: {"acc": epoch_acc, "loss":epoch_loss}})
            if phase == 'train':
                single_train_epoch_loss = format(epoch_loss, '.4f')
                single_train_epoch_acc = format(epoch_acc, '.4f')
            else:
                single_val_epoch_loss = format(epoch_loss,'.4f')
                single_val_epoch_acc = format(epoch_acc,'.4f')

        print(f"\n--------Summary in Epoch {epoch + 1}--------")
        print("--------Train--------")
        print(f'Loss: {single_train_epoch_loss} Acc: {single_train_epoch_acc}')
        print("--------Val--------")
        print(f'Loss: {single_val_epoch_loss} Acc: {single_val_epoch_acc}')

def dataloader(name, path,noise, BATCH_SIZE=64):
    dataset = AudioDataset(path,noise)
    print(f"There are {len(dataset)} samples in the {name} dataset.")

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, pin_memory=True)

    return dataloader, len(dataset)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, input_data):
        predictions = self.model(input_data)
        return predictions

if __name__ == "__main__":
    wandb.init(project=PROJECT, group=GROUP, name=NAME)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device {device}")

    train_loader, train_data_size = dataloader(name="train",path=Train_path, noise="mix")

    test_loader, test_data_size = dataloader(name="test", path=Test_path, noise="none")

    dataloaders_dict = {"train": train_loader, "val": test_loader}

    model = Model()
    model = model.to(device)

    wandb.watch(model)

    print(model)

    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)

    train_model(model, dataloaders_dict, loss_fn, optimizer, device, EPOCHS)

    wandb.finish()





