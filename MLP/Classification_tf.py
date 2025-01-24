'''
1.データセットの読み込みと前処理
・Fashion-MNIST データセットを読み込む
・訓練データ：60000 枚
・テストデータ：10000 枚
・各画像は 28×28 のグレースケール画像（ピクセル値：0～255）
'''
from tensorflow.keras.datasets import fashion_mnist
(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()

'''
2.画像データの前処理
28×28 の画像を 784 次元のベクトルに変換
ピクセル値（0-255）を [0,1] の範囲にスケール変換 し、学習をしやすくする。
'''
x_train = x_train.reshape(-1,784)
x_train = x_train.astype('float32') / 255
x_test = x_test.reshape(-1,784)
x_test = x_test.astype('float32') / 255

'''
3.モデルの定義
Sequential を使用し、モデルを 順番に定義
Dense（全結合層）、Dropout（過学習防止用）、SGD（確率的勾配降下法）をインポート
'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.optimizers import SGD

'''
入力層：784次元、隠れ層：256ニューロン
活性化関数は ReLU（非線形変換を行う）
'''
model = Sequential()
model.add(Dense(256,
               input_dim=784,
               activation='relu'))
'''
ドロップアウト（過学習防止） → 50%のニューロンをランダムに無効化
'''
model.add(Dropout(0.5))

'''
出力層：10クラス（分類問題のため）
活性化関数は Softmax（出力を確率に変換）
'''
model.add(Dense(10,
               activation='softmax'))

'''
4.モデルのコンパイル
・損失関数：
　・sparse_categorical_crossentropy（ラベルが整数の場合に適用）
・オプティマイザー
　・SGD(lr=0.1)（確率的勾配降下法、学習率 0.1）
・評価指標：
　・accuracy（正解率）
'''
learning_rate = 0.1
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=SGD(lr=learning_rate),
    metrics=['accuracy']
)

'''
モデルの構造を出力
'''
model.summary()

%%time
'''
5.モデルの学習
val_loss（検証データの損失）を監視
損失が 5 回以上改善しなかったら学習を停止
verbose=1 で早期終了のログを出力
エポック数：100（学習の繰り返し回数）
ミニバッチサイズ：64（1回の更新で使うデータ数）
'''
from tensorflow.keras.callbacks import EarlyStopping

'''
データを64個に分けて、データの数分学習するのを100回繰り返す
'''
#学習回数、ミニバッチサイズの指定を行う
training_epochs = 100 #学習回数(バッチ学習の繰り返しの全体)
batch_size = 64#ミニバッチのサイズ


early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1
)

'''
訓練データの 20% を検証データとして使用
データをシャッフル
早期終了（early_stopping）を適用
'''
history = model.fit(
    x_train,
    y_train,
    epochs=training_epochs,
    batch_size=batch_size,
    verbose=1,
    validation_split=0.2,
    shuffle=True,
    callbacks=[early_stopping]
)

'''
6.モデルの評価
テストデータでモデルを評価
損失 score[0]、正解率 score[1] を出力
'''
score = model.evaluate(x_test,y_test, verbose=0)
print('Test loss:',score[0])
print('Test accuracy:',score[1])

'''
7.学習結果の可視化
訓練データ (loss)、検証データ (val_loss) の損失の推移を可視化
過学習の傾向を確認
'''
%matplotlib inline
import matplotlib.pyplot as plt

plt.plot(history.history['loss'],
        marker='',
        label='loss(Training)')
plt.plot(history.history['val_loss'],
        marker='',
        label='loss(Validation)')
plt.legend()
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
plt.plot(history.history['accuracy'],
        marker='',
        label='accuracy(Training)')
plt.plot(history.history['val_accuracy'],
        marker='',
        label='accuracy(Vakidation)')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()