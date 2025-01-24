

'''
1.データの読み込みと前処理
'''
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

root = '/data'

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5),(0.5)),
     lambda x:x.view(-1),
    ]
)

f_mnist_train =datasets.FashionMNIST(
    root=root,
    download=True,
    train=True,
    transform=transform)

f_mnist_test = datasets.FashionMNIST(
    root=root,
    download=True,
    train=False,
    transform=transform
)

batch_size = 64
train_dataloader = DataLoader(f_mnist_train,
                             batch_size=batch_size,
                             shuffle=True)

test_dataloader = DataLoader(f_mnist_test,
                             batch_size=batch_size,
                             shuffle=False)

for (x,t) in train_dataloader:
    print(x.shape)
    print(t.shape)
    break

for(x,t) in test_dataloader:
    print(x.shape)
    print(t.shape)
    break
    
print(x)

'''
2.モデルの定義
'''
import torch
import torch.nn as nn

class MLP(nn.Module):
    '''多層パーセプトロン
    Attributes:
     l1(Linear):隠れ層
     l2(Linear):出力層
     d1(Dropout):ドロップアウト
    '''
    def __init__(self,input_dim,hidden_dim,output_dim):
        '''モデルの初期化を行う
         Parameters:
          input_dim(int):入力する1データあたりの値の形状
          hidden_dim(int):隠れ層のユニット
          output_dim(int):出力層のユニット
        '''
        #スーパークラスの__init__()を実行
        super().__init__()
        #隠れ層
        self.fc1 = nn.Linear(input_dim,#入力するデータのサイズ
                             hidden_dim)#隠れ層のニューロン数
        #ドロップアウト
        self.d1 = nn.Dropout(0.5)
        #出力層
        self.fc2 = nn.Linear(hidden_dim,output_dim)##入力するデータのサイズ(前層のニューロンの数),出力層のニューロン数
        
    def forward(self,x):
            '''MLPの順伝播処理を行う
            
            Parameters:
             x(ndarray(float32)):訓練データ、テストデータ
            
            Return(float32):
             出力層からの出力値
            '''
            #レイヤー、活性化関数に前ユニットからの出力を入力する
            x = self.fc1(x)
            x = torch.sigmoid(x)
            x = self.d1(x)
            x = self.fc2(x)#最終出力は活性化関数を適用しない
            return x
        
'''
3.モデルの生成
'''
#使用可能なデバイス(GPUまたはCPU)を取得
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#モデルオブジェクトを生成し、使用可能なデバイスを設定する
model = MLP(784,256,10).to(device)

model#モデルの構造を出力

'''
4.損失とオプティマイザーの生成
'''
import torch.optim

#クロスエントロピー誤差のオブジェクトを生成
criterion = nn.CrossEntropyLoss()
#勾配降下アルゴリズムを使用するオプティマイザーを生成
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)

'''
5.train_step()関数の定義
'''
def train_step(x,t):
    '''バックプロパゲーションによるパラメータの更新を行う
    
    Paremeters:x:訓練データ
               t:正解ラベル
    
    Returns:
        MLPの出力と正解ラベルのクロスエントロピー誤差
    '''
    model.train()#モデルを訓練(学習)モードにする
    preds = model(x)#モデルの出力を取得
    loss = criterion(preds,t)#出力と正解ラベルの誤差から損失を取得
    optimizer.zero_grad()#勾配を0で初期化
    loss.backward()#逆伝播の処理(自動微分による勾配計算)
    optimizer.step()#勾配降下法の更新式を適用してバイアス、重みを更新
    return loss,preds

'''
6.test_step()関数の定義
'''
def test_step(x,t):
    '''テストデータを入力して損失と予測値を返す
    
    Parameters:x:テストデータ
               y:正解ラベル
    Returns:
     MLPの出力と正解ラベルのクロスエントロピー誤差
    '''
    model.eval()#モデルを評価モードにする
    preds =model(x)#モデルの出力を取得
    loss = criterion(preds,t)#出力と正解ラベルの誤差から損失を取得
    
    return loss,preds  

'''
7.学習の進捗を監視し早期終了判定を行うクラス
'''
class EaryStopping:
    def __init__(self,patience=10,verbose=0):
        '''
        Parameters:
         patience(int):監視するエポック数(デフォルトは10)
         verbose(int):早期終了メッセージの出力フラグ
                      出力(1),出力しない(0)
        '''
        #インスタンス変数の初期化
        #監視中のエポック数のカウンターを初期化
        self.epoch = 0
        #比較対象の損失を無限大'inf'で初期化
        self.pre_loss = float('inf')
        #監視対象のエポック数をパラメータで初期化
        self.patience = patience
        #早期終了メッセージの出力フラグをパラメーターで初期化
        self.verbose = verbose
        
    def __call__(self,current_loss):
        '''
        Parameters:
         current_loss(float):1エポック終了後の検証データの損失
        Return:
         True:監視回数の上限までに前エポックの損失を超えた場合
         False:監視回数の上限までに前エポックの損失を超えない場合
        '''
        #前エポックの損失より大きくなった場合
        if self.pre_loss < current_loss:
            self.epoch += 1#カウンターを1増やす
            #監視回数の上限に達した場合
            if self.epoch > self.patience:
                if self.verbose:#早期終了メッセージの出力フラグが1の場合
                    print('early stopping')#メッセージを出力
                return True#学習を終了するTrueを返す
        #目エポックの損失以下の場合
        else:
            self.epoch = 0 #カウンターを0に戻す
            self.pre_loss = current_loss#損失の値を更新
        return False

%%time
'''
8.モデルを使用して学習する
'''
from sklearn.metrics import accuracy_score

#エポック数
epochs = 200
#損失と精度の履歴を保存するためのdictオブジェクト
history = {'loss':[],'accuracy':[],'test_loss':[],'test_accuracy':[]}
#早期終了の判定を行うオブジェクトを生成
ers = EaryStopping(patience=5,verbose=1)#監視対象回数,早期終了時にメッセージを出力
#学習を行う
for epoch in range(epochs):
    train_loss = 0.#訓練1エポックごとの損失を保持する変数
    train_acc = 0.#訓練1エポックごとの精度を保持する変数
    test_loss = 0.#検証1エポックごとの損失を保持する変数
    test_acc = 0.#検証1エポックごとの精度を保持する変数
    
    #1ステップにおける訓練用ミニバッチを使用した学習
    for (x,t) in train_dataloader:
        #torch.Tensorオブジェクトにデバイスを割り当てる
        x,t = x.to(device), t.to(device)
        loss,preds = train_step(x,t)#損失と予測値を取得
        train_loss += loss.item()#ステップごとの損失を加算
        train_acc+= accuracy_score(
            t.tolist(),
            preds.argmax(dim=-1).tolist()
        )#ステップごとの精度を加算
    
    #1ステップにおけるテストデータのミニバッチを使用した評価
    for(x,t) in test_dataloader:
        #torch.Tensorオブジェクトにデバイスを割り当てる
        x,t = x.to(device),t.to(device)
        loss,preds = test_step(x,t)#損失と予測値を取得
        test_loss += loss.item()#ステップごとの損失を加算
        test_acc+= accuracy_score(
            t.tolist(),
            preds.argmax(dim=-1).tolist()
        )#ステップごとの精度を加算
    
    #訓練時の損失の平均を取得
    avg_train_loss = train_loss / len(train_dataloader)
    #訓練時の精度の平均値を取得
    avg_train_acc = train_acc / len(train_dataloader)
    #検証時の損失の平均値を取得
    avg_test_loss = test_loss / len(test_dataloader)
    #検証時の精度の平均値を取得
    avg_test_acc = test_acc / len(test_dataloader)
    
    #訓練データの履歴を保存する
    history['loss'].append(avg_train_loss)
    history['accuracy'].append(avg_train_acc)
    #テストデータの履歴を保存する
    history['test_loss'].append(avg_test_loss)
    history['test_accuracy'].append(avg_test_acc)
    
    #1エポックごとの結果を出力
    if (epoch + 1) % 1 == 0:
        print(
            'epoch({}) train_loss: {:.4} train_acc:{:.4} val_loss: {:.4} val_acc:{:.4}'.format(
             epoch+1,
             avg_train_loss,#訓練データの損失を出力
             avg_train_acc,#訓練データの精度を出力
             avg_test_loss,#テストデータの損失を出力
             avg_test_acc #テストデータの精度を出力
            ))
    
    #テストデータの損失をEarlyStoppingオブジェクトに渡して早期終了を判定
    if ers(avg_test_loss):
        #監視対象のエポックで損失が改善されなけらば学習を終了
        break

'''
9.損失と精度の推移をグラフにする
'''
import matplotlib.pyplot as plt
%matplotlib inline

#損失
plt.plot(history['loss'],
        marker='.',
        label='loss (Training)')
plt.plot(history['test_loss'],
        marker='.',
        label='loss (Test)')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

#精度
plt.plot(history['accuracy'],
        marker='.',
        label = 'accuracy (Training)')
plt.plot(history['test_accuracy'],
        marker='.',
        label='accuracy (Test)')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()     