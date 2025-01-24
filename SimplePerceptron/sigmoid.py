#グラフをインライン（セルの下に直接）表示するためのマジックコマンドです。
%matplotlib inline

#2. ライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt

#3. シグモイド関数の定義
def sigmoid(x):
    '''
    値を0～1の範囲に変換する関数
    '''
    return 1 / (1 + np.exp(-x))

x = np.arange(-5.0,5.0,0.01)
'''
np.arrange(start,stop,step)：startからstopまでstep間隔で数列を作成
'''

y = sigmoid(x)
'''
xの各値をsigmoid(x)に通し、対応するyを計算
'''

plt.plot(x,y)
plt.grid(True)
plt.show()
'''
X軸にx、y軸にyの折れ線グラフを描画
'''