'''
1.データの読み込み
'''
import os
import json

# コーパスのディレクトリを設定
file_path = './json/rest1046/'
# ファイルの一覧を取得
file_dir = os.listdir(file_path)
# 人間の発話を保持するリスト
utterance_txt = []
# システムの応答を保持するリスト
system_txt = []
# 正解ラベルを保持するリスト
label = []

# ファイルごとに対話データを整形する
for file in file_dir:
    # JSONファイルの読み込み
    r = open(file_path + file, 'r', encoding='utf-8')
    json_data = json.load(r)
        
    # 発話データ配列から発話テキストと破綻かどうかの正解データを抽出
    for turn in json_data['turns']:
        turn_index = turn['turn-index'] # turn-indexキー(対話のインデックス)
        speaker = turn['speaker']       # speakerキー("U"人間、"S"システム)
        utterance = turn['utterance']   # utteranceキー(発話テキスト)
        
        # 先頭行(システムの冒頭の発話)以外を処理
        if turn_index != 0:
            # 人間の発話（質問）のテキストを抽出
            if speaker == 'U':
                #u_text = ''
                u_text = utterance

            # システムの応答内容が破綻かどうかを抽出
            else:
                a = ''
                sys = turn['utterance'] # システムの発話（応答）を抽出
                t = turn['annotations'][0] # １つ目のアノテーションを抽出                
                a = t['breakdown']      # アノテーションのフラグを抽出
                if a == 'O':            # O（破綻していない）を0で置換
                    val = 0
                elif a == 'T':          # T（破綻していないが違和感がある）を1で置換
                    val = 1
                else:                   # 上記以外のX（破綻している）は2で置換
                    val = 2
                # 人間の発話をリストに追加
                utterance_txt.append(u_text)
                # システムの応答をリストに追加
                system_txt.append(sys)
                # 正解ラベルをリストに追加
                label.append(str(val))

'''
2. 読み込んだデータのサイズを出力
'''
print(len(utterance_txt)) # 人間の発話のサイズ
print(len(system_txt))    # システムの応答のサイズ
print(len(label))         # 正解ラベルのサイズ

'''
3.utterance_textに格納されたテキストを出力する
'''
utterance_txt

'''
4. システムの応答を出力
'''
system_txt

'''
5. 正解ラベルを出力
'''
label

'''
6. 人間の発話、システムの応答、正解ラベルをデータフレームにまとめる
''' 
import pandas as pd
df = pd.DataFrame({'utterance_txt' : utterance_txt,
                   'system_txt' : system_txt,
                   'label' : label}
                 )
df   # 出力

'''
7. 破綻していない(0)、破綻していないが違和感(1)、破綻(2)をカウント
''' 
df['label'].value_counts()

'''
8. 形態素への分解
言葉を、品詞、名詞、動詞、助詞、助動詞といった節単位に分解する
''' 
from janome.tokenizer import Tokenizer # janomeのパッケージをインポート
import re                              # 正規表現ライブラリ

def parse(utterance_txt):
    '''
    分かち書きを行って形態素に分解する
    '''
    t = Tokenizer()                     # Tokenizerクラスのオブジェクトを生成
    separation_tmp = []                 # 形態素を一時保存するリスト
    # 形態素に分解
    for row in utterance_txt:
        # リストから発話テキストの部分を抽出して形態素解析を実行
        tokens = t.tokenize(row)
        # 形態素の見出しの部分を取得してseparation_tmpに追加
        separation_tmp.append(
            [token.surface for token in tokens if (
                not re.match('記号', token.part_of_speech)             # 記号を除外
                and (not re.match('助詞', token.part_of_speech))       # 助詞を除外
                and (not re.match('助動詞', token.part_of_speech))     # 助動詞を除外
                )
             ])
        # 空の要素があれば取り除く
        while separation_tmp.count('') > 0:
            separation_tmp.remove('')
    return separation_tmp

# 人間の発話を形態素に分解する
df['utterance_token'] = parse(df['utterance_txt'])
# システムの応答を形態素に分解する
df['system_token'] = parse(df['system_txt'])

# 形態素への分解後のデータフレームを出力
df

'''
9. 発話と応答それぞれの形態素の数をデータフレームに登録する
''' 
df['u_token_len'] = df['utterance_token'].apply(len)
df['s_token_len'] = df['system_token'].apply(len)
df

'''
10. 単語の出現回数を記録して辞書を作成
'''
from collections import Counter   # カウント処理のためのライブラリ
import itertools                   # イテレーションのためのライブラリ

# {単語：出現回数}の辞書を作成
def makedictionary(data):
    return Counter(itertools.chain(* data))

'''
11. 単語を出現回数順(降順)に並べ替えて連番をふる関数
'''
def update_word_dictionary(worddic):
    word_list = []
    word_dic = {}
    # most_common()で出現回数順に要素を取得しword_listに追加
    for w in worddic.most_common():
        word_list.append(w[0])

    # 頻度順に並べた単語をキーに、1から始まる連番を値に設定
    for i, word in enumerate(word_list, start=1):
        word_dic.update({word: i})
        
    return word_dic

'''
12. 単語を出現頻度の数値に置き替える関数
'''
def bagOfWords(word_dic, token):
    return [[ word_dic[word] for word in sp] for sp in token]

'''
13.発話を{単語：出現回数}の辞書にする
'''
utter_word_frequency = makedictionary(df['utterance_token'])
utter_word_frequency

'''
14.応答を{単語：出現回数}の辞書にする
'''
system_word_frequency = makedictionary(df['system_token'])
system_word_frequency

'''
15.発話の単語辞書を単語の頻度順（降順）で並べ替えて連番を割り当てる
'''
utter_word_dic = update_word_dictionary(utter_word_frequency)
utter_word_dic

'''
16.応答の単語辞書を単語の頻度順（降順）で並べ替えて連番を割り当てる
'''
system_word_dic = update_word_dictionary(system_word_frequency)
system_word_dic

'''
17. 辞書のサイズを変数に登録
'''
utter_dic_size = len(utter_word_dic)
system_dic_size = len(system_word_dic)
print(utter_dic_size)
print(system_dic_size)

'''
18. 単語を出現頻度順の数値に置き換える
'''
train_utter = bagOfWords(utter_word_dic, df['utterance_token'])
train_system = bagOfWords(system_word_dic, df['system_token'])

'''
19. 数値に置き換えた発話をを出力
例：[335, 471, 7, 14]の場合、言葉は4つの品詞から構成されていて、それぞれ頻出頻度が、
335位と471位と7位と14位の単語で構成されている
'''
train_utter

'''
20.数値に置き換えた応答を出力 
'''
train_system

'''
21. 発話と応答それぞれの形態素の最大数を取得
'''
UTTER_MAX_SIZE = len(sorted(train_utter, key=len, reverse=True)[0])
SYSTEM_MAX_SIZE = len(sorted(train_system, key=len, reverse=True)[0])
print(UTTER_MAX_SIZE)
print(SYSTEM_MAX_SIZE)

'''
22. 単語データの配列を同一のサイズに揃える関数
'''
from tensorflow.keras.preprocessing import sequence

def padding_sequences(data, max_len):
    '''最長のサイズになるまでゼロを埋め込む
    Parameters:
      data(array): 操作対象の配列
      max_len(int): 配列のサイズ
    '''
    return sequence.pad_sequences(
        data, maxlen=max_len, padding='post',value=0.0)
    
'''
23. 発話の単語配列のサイズを最長サイズに合わせる
'''
train_U = padding_sequences(train_utter, UTTER_MAX_SIZE)
print(train_U.shape)
print(train_U)

'''
24. 応答の単語配列のサイズを最長サイズに合わせる
'''
train_S = padding_sequences(train_system, SYSTEM_MAX_SIZE)
print(train_S.shape)
print(train_S)

'''
25. 訓練データとテストデータに分割する
'''
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# データの先頭から80パーセントを訓練データ
# 残り20パーセントをテストデータに分割する
train_df, val_df = train_test_split(
    df, train_size=0.8, shuffle=False)

# 訓練データとテストデータのデータ数を取得
train_df_num = train_df.shape[0]
val_df_num = val_df.shape[0]

# 訓練データのdictオブジェクトを作成
trainX = {
    # 人間の発話
    'utterance': train_U[:train_df_num],
    # システムの応答
    'system': train_S[:train_df_num],    
    # 人間の発話の形態素の数(int)
    'u_token_len': np.array(train_df[['u_token_len']]),
    # システムの応答の形態素の数(int)
    's_token_len': np.array(train_df[['s_token_len']])
}

# 正解ラベルをOne-hot表現のndarrayにする
trainY = to_categorical(train_df['label'], 3)

print((trainX['utterance'].shape))
print((trainX['system'].shape))
print(trainY.shape)

# テストデータのdictオブジェクトを作成
testX = {
    # 人間の発話
    'utterance': train_U[train_df_num:],
    # システムの応答
    'system': train_S[train_df_num:],
    
    # 人間の発話の形態素の数(int)
    'u_token_len': np.array(val_df[['u_token_len']]),
    # システムの応答の形態素の数(int)
    's_token_len': np.array(val_df[['s_token_len']])
}

# 正解ラベルをOne-hot表現にする
testY = to_categorical(val_df['label'], 3)

print((testX['utterance'].shape))
print((testX['system'].shape))
print(testY.shape)

y_test_label = np.array(val_df['label'], dtype= np.float_ )

'''
26. モデルを構築する関数
'''
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Conv1D, MaxPooling1D
from tensorflow.keras.layers import concatenate, Flatten, Dense, Dropout
from tensorflow.keras import models, layers, optimizers, regularizers
#Conv1D, MaxPooling1Dは音声信号、自然言語のデータに対してフィルターを行うクラス
#⇒フィルターのサイズが2の場合は、シーケンス方向の2次元ベクトルの重みをスライドさせてフィルターを行う
#Conv2D, MaxPooling2Dは縦と横に対してフィルターを行う画像のフィルターを行うクラス
#⇒フィルターのサイズが2の場合は、縦方向2×横方向2=4の重みをスライドさせてフィルターを行う

def create_RNN(model_num):
    """モデルを生成する

    Parameters: model_num(int):
      モデルの番号
    Returns:
      Modelオブジェクト
    """
    
    rnn_weight_decay = 0.001
    cnn_weight_decay = 0.01

    ## ------入力層------
    # 人間の発話:ユニット数は単語配列の最長サイズと同じ
    utterance = Input(shape=(UTTER_MAX_SIZE,), name='utterance')
    # システムの応答:ユニット数は単語配列の最長サイズと同じ
    system = Input(shape=(SYSTEM_MAX_SIZE,), name='system')
    # 人間の発話の単語数:ユニット数は1
    u_token_len = Input(shape=[1], name="u_token_len")
    # システムの応答の単語数:ユニット数は1
    s_token_len = Input(shape=[1], name="s_token_len")
    
    # ------Embedding層------
    #単語IDを高次元の実数ベクトル（埋め込みベクトル）に変換
    # 人間の発話: 入力は単語の総数+100、出力の次元数は64
    emb_utterance = Embedding(
        input_dim=utter_dic_size+100,  # 発話の単語数+100
        output_dim=64,                 # 各単語を64次元の埋め込みベクトルに変換(ニューロンの数：64)
        )(utterance)
    # システムの応答: 入力は単語の総数+100、出力の次元数は128
    emb_system = Embedding(
        input_dim=system_dic_size+100, # 応答の単語数+100
        output_dim=64                  # 各単語を64次元の埋め込みベクトルに変換(ニューロンの数：64)
        )(system)
    # 人間の発話の単語数のEmbedding
    emb_u_len = Embedding(
        input_dim=UTTER_MAX_SIZE+1,    # 発話の最大単語数（形態素数）
        output_dim=5                   # 発話の長さを5次元のベクトルに変換
        )(u_token_len)
    # システムの応答の単語数のEmbedding
    emb_s_len = Embedding(
        input_dim=SYSTEM_MAX_SIZE+1,   # 発話の最大単語数（形態素数）
        output_dim=5                   # 発話の長さを5次元のベクトルに変換
        )(s_token_len)
    
    #-----RNN-----
    #model_num が 8 未満の場合に、以下のコードブロックを実行
    if model_num < 8:
        # 人間の発話を入力
        #ニューロンの数は、64
        #入力に対して50%のドロップアウトを適用
        # LSTMユニット×64 正則化あり
        #rnn_weight_decay：正則化の係数
        #正則化=原点からの距離が大きい重みに対して罰則を与える
        layer1_1 = LSTM(64, return_sequences=True, dropout=0.5,
                        kernel_regularizer=regularizers.l2(rnn_weight_decay))(emb_utterance)
        # LSTMユニット×128 正則化あり
        layer1_final = LSTM(128, return_sequences=False, dropout=0.5,
                            kernel_regularizer=regularizers.l2(rnn_weight_decay))(layer1_1)
        
        # システムの応答を入力
        # LSTMユニット×64 正則化あり
        layer2_1 = LSTM(64, return_sequences=True, dropout=0.1,
                        kernel_regularizer=regularizers.l2(rnn_weight_decay))(emb_system)
        # LSTMユニット×128 正則化あり
        layer2_final = LSTM(128, return_sequences=False, dropout=0.1,
                            kernel_regularizer=regularizers.l2(rnn_weight_decay))(layer2_1)
    #-----CNN-----
    else:
        # 人間の発話を入力
        # 畳み込み層1 フィルター数64 正則化あり
        #入力データの行列に対して、サイズ5のフィルターをシーケンス方向に畳み込みを行う
        #64個の異なるフィルターデータを使用してフィルターを行う
        #メモ：64個(64チャネル)を再度演算する場合は、64チャネルの各々のチャネル分重みが用意される、画像も同様
        layer1_1 = Conv1D(filters=64,kernel_size=(5), padding='same', activation='relu',
                        kernel_regularizer=regularizers.l2(cnn_weight_decay))(emb_utterance)
        # 畳み込み層2 フィルター数128 正則化あり
        layer1_2 = Conv1D(filters=128,kernel_size=(5), padding='same', activation='relu',
                          kernel_regularizer=regularizers.l2(cnn_weight_decay))(layer1_1)
            
        # プーリング層、ドロップアウト
        layer1_3 = MaxPooling1D(pool_size=2, strides=2, padding='same')(layer1_2)
        layer1_final = Dropout(0.5)(layer1_3)
            
        # システムの応答を入力
        # 畳み込み層1 フィルター数64 正則化あり
        layer2_1 = Conv1D(filters=64,kernel_size=(5), padding='same', activation='relu',
                        kernel_regularizer=regularizers.l2(cnn_weight_decay))(emb_system)

        # 畳み込み層2 フィルター数128 正則化あり
        layer2_2 = Conv1D(filters=128,kernel_size=(5), padding='same', activation='relu',
                             kernel_regularizer=regularizers.l2(cnn_weight_decay))(layer2_1)
        # プーリング層、ドロップアウト
        layer2_3 = MaxPooling1D(pool_size=2, strides=2, padding='same')(layer2_2)
        layer2_final = Dropout(0.5)(layer2_3)
            
    #-----全結合-----
    # ------全結合層------
    main_l = concatenate([
        Flatten()(emb_u_len),    # 人間の発話の単語数のEmbedding
        Flatten()(emb_s_len),    # システム応答の単語数のEmbedding
         # 人間の発話のRNNまたはCNNからの出力
        #RNNは最後の時刻の出力だけがフラット化される
        #CNNの場合は、人間とシステムの２つの時刻のデータを畳み込み層で分けているため、ここで結合される
        Flatten()(layer1_final),
        # システム応答のRNNまたはCNNからの出力
        #RNNは最後の時刻の出力だけがフラット化される
        #CNNの場合は、人間とシステムの２つの時刻のデータを畳み込み層で分けているため、ここで結合される
        Flatten()(layer2_final)  
    ])
    
    # ------512、256、128ユニットの層を追加------
    main_l = Dropout(0.5)(
        Dense(512,kernel_initializer='normal',activation='relu')(main_l))
    main_l = Dropout(0.5)(
        Dense(256,kernel_initializer='normal',activation='relu')(main_l))
    main_l = Dropout(0.5)(
        Dense(128,kernel_initializer='normal',activation='relu')(main_l))
    
    # ------出力層(3ユニット)------
    output = Dense(units=3,              # 出力層のニューロン数＝3
                   activation='softmax'  # 活性化はソフトマックス関数
                   )(main_l)


    # Modelオブジェクトの生成
    model = models.Model(
        # 入力層はマルチ入力モデルなのでリストにする
        inputs=[utterance, system, u_token_len, s_token_len],
        # 出力層
        outputs=output
    )
    
    return model 

'''
27. アンサンブルを実行する関数
'''
from scipy.stats import mode#配列内の最頻値を計算するために使用

#models=予測に使用するモデルのリスト、X=入力データ、data_num = データのサンプル数
def ensemble_majority(models, X, data_num):
    #モデルごとの予測結果を格納
    #data_num=データのサンプル数、len(models)=モデルの数
    pred_labels = np.zeros((data_num,     
                            len(models))) 
    #models の中の各モデルについて予測
    for i, model in enumerate(models):
        #各サンプルについて、最も高い確率を持つクラスのインデックス（ラベル）を取得
        pred_labels[:, i] = np.argmax(model.predict(X), axis=1)
    # 多数決による最終予測ラベルの決定
    return np.ravel(mode(pred_labels, axis=1)[0])

'''
28. 複数のモデルで学習し、アンサンブルによる評価を行う関数
'''
#RNNモデルを8個　CNNモデルを7個用意して各々モデル1個づつにデータセットで学習を行い、検証データで各々のモデル15個に対して予測を行い、予測したクラスのインデックスが最も多かったデータを予測値とする
import math
import numpy as np
from sklearn.metrics import accuracy_score
from keras.callbacks import LearningRateScheduler
from keras.callbacks import History
from tensorflow.keras import models, layers, optimizers, regularizers

def train(trainX, trainY, testX, testY, y_test_label):
    #trainX=訓練データの特徴量　trainY：訓練データのラベル　testX：テストデータの特徴ラベル　testY：テストデータのラベル

    n_estimators = 15#学習するモデルの数
    batch_size = 64 #学習時のバッチサイズ
    epoch = 3  #各モデルの学習におけるエポック数   
    models = []   #学習したモデルを格納するリスト
    #hists=各モデルの学習履歴を保存するリスト ensemble_test = アンサンブル評価の精度
    history_all = {"hists":[], "ensemble_test":[]}
    model_predict = np.zeros((val_df_num, #各モデルの予測結果を格納するための配列
                             n_estimators))
    
    #15個のモデルを順次学習し、それぞれの学習後にアンサンブル評価
    for i in range(n_estimators):
        print('Model',i+1)#現在学習中のモデル番号
        train_model = create_RNN(i)#i の値に応じて異なるモデル（RNNやCNN）を生成する
        train_model.compile(optimizer=optimizers.Adam(lr=0.001),#学習率0.001のAdamオプティマイザを使用
                            loss='categorical_crossentropy',#多クラス分類の損失関数
                            metrics=["acc"])#学習中に精度をモニタリング
        models.append(train_model)#学習したモデルをリストに追加
        hist = History()#学習の履歴を記録
        
        #モデルを訓練
        train_model.fit(
            trainX, trainY,#訓練データとそのラベル       
            batch_size=batch_size, #バッチサイズ
            epochs=epoch,#エポック数    
            verbose=1, #学習の進捗情報を表示           
            callbacks=[hist]#学習履歴を記録するコールバック
            )

        
        #学習済みモデルでテストデータに対する予測
        #各サンプルに対して最も高い確率を持つクラスのインデックスを取得
        model_predict[:, i] = np.argmax(train_model.predict(testX),
                                       axis=-1) 

        #現在のモデルの学習履歴を保存
        history_all['hists'].append(hist.history)
        
        #複数のモデルからの予測結果を集約し、多数決によって最終的な予測ラベルを決定
        ensemble_test_pred = ensemble_majority(models, testX, val_df_num)
        
        #アンサンブルによる予測結果と正解ラベルを比較し、精度（Accuracy）を計算
        ensemble_test_acc = accuracy_score(y_test_label, ensemble_test_pred)
        
        #アンサンブル評価の精度を保存
        history_all['ensemble_test'].append(ensemble_test_acc)
        
        #アンサンブルによるモデルの数が増えていく過程の精度
        print('Current Ensemble Test Accuracy : ', ensemble_test_acc)
        
%%time
# アンサンブルを実行
train(trainX, trainY, testX, testY, y_test_label)