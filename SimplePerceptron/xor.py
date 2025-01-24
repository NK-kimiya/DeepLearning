#XOR問題の学習

#1・必要なライブラリのインポート
import numpy as np
import torch

'''
2.XOR問題のデータセット作成
入力(x1,x2)|出力(y)
　　(0,0)  |　0
　　(0,1)  |　1
　　(1,0)  |　1
　　(1,1)  |　0

'''
train = np.array([[0,0],[0,1],[1,0],[1,1]])#入力データ
label = np.array([[0],[1],[1],[0]])#出力データ

'''
3.Numpy配列をPyTorchテンソルに変換
'''
train_x = torch.Tensor(train)
train_y = torch.Tensor(label)

import torch.nn as nn

'''
4.MLP(多層パーセプトロン)モデルの定義
'''
class MLP(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super().__init__()#親クラスのコンストラクタを呼び出す
        
        #入力層→隠れ層の全結合
        self.fc1 = nn.Linear(input_dim,
                             hidden_dim)
        
        #隠れ層→出力層の全結合
        self.fc2 = nn.Linear(hidden_dim,#隠れ層から出力層への全結合を定義
                             output_dim)
    
    #順伝播の計算
    def forward(self,x):
        #入力層→隠れ層
        x = self.fc1(x)
        #シグモイド関数で非線形変換
        x = torch.sigmoid(x)
        #隠れ層→出力層
        x = self.fc2(x)
        #シグモイド関数でfc2層の出力
        x = torch.sigmoid(x)
        
        return x

'''
5.モデルの作成&GPU/CPUの設定
'''
#GPUが使える環境なら"cuda"を、なければ"cpu"を選択
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#作成したモデルを GPU または CPU に移動。
model = MLP(2,2,1).to(device)
model

'''
6.トレーニングの準備
'''
def train_step(x,t):#モデルの1回のトレーニングステップ
    '''
    引数x：入力データ
    引数t：正解データ
    '''
    #モデルを学習モードにする
    model.train()
    #入力データxをモデルに通して予測値を取得
    outputs = model(x)
    #予測値と正解値の損失を計算
    loss = criterion(outputs,t)
    #勾配の初期化
    optimizer.zero_grad()
    # 誤差逆伝播（勾配を計算）
    loss.backward()
    #計算された勾配に基づいてモデルのパラメータを更新
    optimizer.step()
    #トレーニングステップでの損失値
    return loss

'''
7.トレーニングの実行
'''
#エポック数の指定
epochs = 4000
for epoch in range(epochs):#エポックの数トレーニングをする
    # 各エポックの損失を累積する変数
    epoch_loss = 0.
    #データをGPUまたはCPUに転送
    train_x,train_y = train_x.to(device),train_y.to(device)
    #1回の学習ステップを実行
    loss = train_step(train_x,train_y)
    #損失を記録
    epoch_loss += loss.item()
    #100エポックごとに損失を出力
    if (epoch + 1) % 100  == 0:
        print('epoch({}) loss:{:.4f}'.format(epoch+1,epoch_loss))

'''
8.学習後のモデルで予測
'''
#学習済みモデルで順伝播
outputs = model(train_x)
print(outputs)

'''
9.予測結果を閾値0.5で丸めて0or1に変換
0.5以上なら1、それ以外は0
'''                                                            　
print((outputs.to('cpu').detach().numpy().copy() > 0.5).astype(np.int32))
