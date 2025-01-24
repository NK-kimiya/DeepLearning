'''
ニューラルネットワークの 順伝播（フォワードプロパゲーション）
'''

'''
1.必要なライブラリのインポート
'''
import numpy as np

'''
2.活性化関数の定義
'''
#シグモイド関数→S字型の曲線を持ち、値を 0 から 1 に圧縮 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#恒等関数→入力をそのまま出力する
def identity_function(x):
    return x

'''
3.重みとバイアスの初期化
'''
def init_paraml():
    #各層の重み・バイアスを格納
    parameters = {}
    #第1層　(入力→隠れ層1)：入力は2次元、出力は隠れ層1のユニット数3
    parameters['W1'] = np.array([[0.1,0.3,0.5],
                                [0.2,0.4,0.6]])
    
    #１層目のバイアス：出力3つにバイアスを加える
    parameters['b1'] = np.array([0.1,0.2,0.3])
    
    #2層目の重み行列：入力が3次元、出力が2ユニット
    parameters['W2'] = np.array([[0.1,0.4],
                                [0.2,0.5],
                                [0.3,0.6]])
    
    #2層目のバイアス：出力2つにバイアスを加える
    parameters['b2'] = np.array([0.1,0.2])
    
    #3層目の重み行列：入力が2次元、出力が2ユニット
    parameters['W3'] = np.array([[0.1,0.3],
                                [0.2,0.4]])
    
    #3層目のバイアス：出力2つにバイアスを加える                    
    parameters['b3'] = np.array([0.1,0.2])
    
    return parameters

'''
4.各層の重みとバイアスを取り出す
'''
parameters = init_paraml()
W1,W2,W3 = parameters['W1'],parameters['W2'],parameters['W3']#param辞書から各層の重みを取り出す
b1,b2,b3 = parameters['b1'],parameters['b2'],parameters['b3']#param辞書から各層のバイアスを取り出す

'''
5.順伝播の計算
'''
#入力xと重みW1の行列積にバイアスを加える
al = np.dot(x,W1) + b1
print(al)
#シグモイド関数を適用し、非線形変換を加える
z1 = sigmoid(al)
print(zl)#計算したalにシグモイド関数を適用

#隠れ層1の出力×重み＋バイアス
a2 = np.dot(z1,W2) + b2
print(a2)
#シグモイド関数を適用し、非線形変換を加える。
z2 = sigmoid(a2)
print(z2)

#隠れ層3の出力×重み＋バイアス
a3 = np.dot(z2,W3) + b3
print(a3)
#恒等関数を適用し、出力
y = identity_function(a3)
print(y)