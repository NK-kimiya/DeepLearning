'''
1. データの読み込みと前処理
'''
import os
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ダウンロード先のディレクトリ
root = './data'

# トランスフォーマーオブジェクトを生成
transform_train = transforms.Compose(
    [transforms.RandomHorizontalFlip(0.2), # 0.2の確率で水平方向反転
     transforms.RandomRotation(15), # 15度の範囲でランダムに回転
     transforms.ColorJitter(brightness=0.3,  # 明度の変化係数
                            saturation=0.3), # 彩度の変化係数
     transforms.ToTensor(), # Tensorオブジェクトに変換
     transforms.Normalize((0.5), (0.5)) # 平均0.5、標準偏差0.5で正規化
     ])

transform_val = transforms.Compose(
     [transforms.ToTensor(), # Tensorオブジェクトに変換
     transforms.Normalize((0.5), (0.5)) # 平均0.5、標準偏差0.5で正規化
     ])

# 訓練用データの読み込み(60000セット)
f_mnist_train = datasets.CIFAR10(
    root=root,     # データの保存先のディレクトリ
    download=True, # ダウンロードを許可
    train=True,    # 訓練データを指定
    transform=transform_train) # トランスフォーマーオブジェクトを指定

# テスト用データの読み込み(10000セット)
f_mnist_test = datasets.CIFAR10(
    root=root,     # データの保存先のディレクトリ
    download=True, # ダウンロードを許可
    train=False,   # テストデータを指定
    transform=transform_val) # トランスフォーマーオブジェクトを指定

# ミニバッチのサイズ
batch_size = 64
# 訓練用のデータローダー
train_dataloader = DataLoader(f_mnist_train, # 訓練データ
                              batch_size=batch_size, # ミニバッチのサイズ
                              shuffle=True) # シャッフルして抽出
# テスト用のデータローダー
test_dataloader = DataLoader(f_mnist_test, # テストデータ
                             batch_size=batch_size, # ミニバッチのサイズ
                             shuffle=False) # シャッフルして抽出

# データローダーが返すミニバッチの先頭データの形状を出力
for (x, t) in train_dataloader: # 訓練データ
    print(x.shape)
    print(t.shape)
    break

for (x, t) in test_dataloader: # テストデータ
    print(x.shape)
    print(t.shape)
    break

'''
2. モデルの定義
'''
import torch.nn as nn

class CNN(nn.Module):
    '''畳み込みニューラルネットワーク
    
        Attributes:
          conv1, conv2, conv3, conv4, conv5, conv6 : 畳み込み層
          bn1, bn2, bn3, bn4, bn5, bn6 : 正規化
          pool1, pool2, pool3 : プーリング層
          dropout1, dropout2, dropout3, dropout4 : ドロップアウト
          fc1, fc2 : 全結合層
    '''
    def __init__(self):
        '''モデルの初期化を行う
          
        '''
        # スーパークラスの__init__()を実行
        super().__init__()

        # 第1層: 畳み込み層1
        # (3,3,32) -> (32,32,32)
        self.conv1 = nn.Conv2d(in_channels=3,   # 入力チャネル数
                               out_channels=32, # 出力チャネル数
                               kernel_size=3,   # フィルターサイズ
                               padding=True,    # パディングを行う
                               padding_mode='zeros')
        # 正規化
        self.bn1 = torch.nn.BatchNorm2d(32)

        # 第2層: 畳み込み層2
        # (32,32,32) ->(32,32,32)
        self.conv2 = nn.Conv2d(in_channels=32,  # 入力チャネル数
                               out_channels=32, # 出力チャネル数
                               kernel_size=3,   # フィルターサイズ
                               padding=True,    # パディングを行う
                               padding_mode='zeros')
        # 正規化
        self.bn2 = torch.nn.BatchNorm2d(32)
       
        # 第3層: プーリング層1
        # (32,32,32) -> (32,16,16)
        self.pool1 = nn.MaxPool2d(2, 2)
        # ドロップアウト1: 20%
        self.dropout1 = nn.Dropout2d(0.2)

        # 第4層: 畳み込み層3
        # (32,16,16) ->(64,16,16)
        self.conv3 = nn.Conv2d(in_channels=32,  # 入力チャネル数
                               out_channels=64, # 出力チャネル数
                               kernel_size=3,   # フィルターサイズ
                               padding=True,    # パディングを行う
                               padding_mode='zeros')
        # 正規化
        self.bn3 = torch.nn.BatchNorm2d(64)

        # 第5層: 畳み込み層4
        # (64,16,16) ->(64,16,16)
        self.conv4 = nn.Conv2d(in_channels=64,  # 入力チャネル数
                               out_channels=64, # 出力チャネル数
                               kernel_size=3,   # フィルターサイズ
                               padding=True,    # パディングを行う
                               padding_mode='zeros')
        # 正規化
        self.bn4 = torch.nn.BatchNorm2d(64)

        # 第6層: プーリング層2
        # (64,16,16) -> (64,8,8)
        self.pool2 = nn.MaxPool2d(2, 2)
        # ドロップアウト2: 30%
        self.dropout2 = nn.Dropout2d(0.3)

       # 第7層: 畳み込み層5
       # (64,8,8) ->(128,8,8)
        self.conv5 = nn.Conv2d(in_channels=64,  # 入力チャネル数
                               out_channels=128,# 出力チャネル数
                               kernel_size=3,   # フィルターサイズ
                               padding=True,    # パディングを行う
                               padding_mode='zeros')
        # 正規化
        self.bn5 = torch.nn.BatchNorm2d(128)

        # 第8層: 畳み込み層6
        # (128,8,8) ->(128,8,8)
        self.conv6 = nn.Conv2d(in_channels=128, # 入力チャネル数
                               out_channels=128,# 出力チャネル数
                               kernel_size=3,   # フィルターサイズ
                               padding=True,    # パディングを行う
                               padding_mode='zeros')
        # 正規化
        self.bn6 = torch.nn.BatchNorm2d(128)

        # 第9層: プーリング層3
        # (128,8,8) -> (128,4,4)
        self.pool3 = nn.MaxPool2d(2, 2)
        # ドロップアウト3: 40%
        self.dropout3 = nn.Dropout2d(0.4)

        # 第10層: 全結合層1
        # (128,4,4) -> (2048,128)
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        # ドロップアウト4: 40%
        self.dropout4 = nn.Dropout2d(0.4)

        # 第11層: 出力層
        # (2048,128) -> (128,10)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        '''MLPの順伝播処理を行う
        
        Parameters:
          x(ndarray(float32)):訓練データ、または検証データ
          
        Returns(float32):
          出力層からの出力値    
        '''
        x = torch.relu(self.conv1(x))
        x = self.bn1(x)
        x = torch.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = torch.relu(self.conv3(x))
        x = self.bn3(x)
        x = torch.relu(self.conv4(x))
        x = self.bn4(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x = torch.relu(self.conv5(x))
        x = self.bn5(x)
        x = torch.relu(self.conv6(x))
        x = self.bn6(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = x.view(-1, 128 * 4 * 4)  # フラット化
        x = torch.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)
        return x

'''
3. モデルの生成
'''
import torch

# 使用可能なデバイス(CPUまたはGPU）を取得する
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# モデルオブジェクトを生成し、使用可能なデバイスを設定する
model = CNN().to(device)

model # モデルの構造を出力

'''
4. 損失関数とオプティマイザーの生成
'''
import torch.optim

# クロスエントロピー誤差のオブジェクトを生成
criterion = nn.CrossEntropyLoss()
# 勾配降下アルゴリズムを使用するオプティマイザーを生成
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,             # 学習率
    weight_decay=0.0001)  # L2正則化のハイパーパラメーター 

'''
5. 勾配降下アルゴリズムによるパラメーターの更新処理
'''
def train_step(x, t):
    '''バックプロパゲーションによるパラメーター更新を行う
    
    Parameters: x: 訓練データ
                t: 正解ラベル
                
    Returns:
      MLPの出力と正解ラベルのクロスエントロピー誤差
    '''
    model.train()    # モデルを訓練(学習)モードにする
    preds = model(x) # モデルの出力を取得
    loss = criterion(preds, t) # 出力と正解ラベルの誤差から損失を取得
    optimizer.zero_grad() # 勾配を0で初期化（累積してしまうため）
    loss.backward()  # 逆伝播の処理(自動微分による勾配計算)
    optimizer.step() # 勾配降下法の更新式を適用してバイアス、重みを更新

    return loss, preds 

'''
6. モデルの評価を行う関数
'''
def test_step(x, t):
    '''テストデータを入力して損失と予測値を返す
    
    Parameters: x: テストデータ
                t: 正解ラベル
    Returns:
      MLPの出力と正解ラベルのクロスエントロピー誤差
    '''
    model.eval()     # モデルを評価モードにする
    preds = model(x) # モデルの出力を取得
    loss = criterion(preds, t) # 出力と正解ラベルの誤差から損失を取得

    return loss, preds 

%%time
'''
6.モデルを使用して学習する
'''
from sklearn.metrics import accuracy_score

epochs = 120
history = {'loss':[],'accuracy':[],'test_loss':[],'test_accuracy':[]}
ers = EarlyStopping(patience=10,
                   verbose=1)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.1,
)