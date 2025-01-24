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
5.RNNモデルの構築
入力層：前処理を行い形態要素を分解、品詞や単語を数値で表す。
Embedding層(形態要素に対して各品詞をベクトルで表し行列にデータを変換し、トークン数もベクトルに変換)
Reccurrent層(Embedding層の行列に対して重みを加えた伝播を行う)
全結合層(Reccurrent層の出力の行列を１次元にする)
'''
'''
25. RNNモデルの構築
'''
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, concatenate, GRU, Embedding, Flatten, Dropout
from tensorflow.keras import models, layers, optimizers, regularizers

## ------入力層------
##入力層のニューロンの数の定義(形態要素で品詞に分けたとき最大の品詞の数)
#入力データの形状：（例: [1, 2, 3, 4, 5, 6, 7, 8,0,0......]）
utterance = Input(shape=(UTTER_MAX_SIZE,), name='utterance')
system = Input(shape=(SYSTEM_MAX_SIZE,), name='system')

'''
データは、人間の発話とシステムの発話の２種類から構成されているため、変数を２つ用意する
'''
# 人間の発話の単語数:ユニット数は1
u_token_len = Input(shape=[1], name="u_token_len")#単語や形態素の数
# システムの応答の単語数:ユニット数は1
s_token_len = Input(shape=[1], name="s_token_len")#単語や形態素の数

#ユーザーの会話のデータから
# ------Embedding層------形態要素で分解したデータの各品詞(単語)をベクトルで表す+分解した品詞の数をベクトルで表す

# 人間の発話: 入力は単語の総数+100、出力の次元数は128
emb_utterance = Embedding(
    input_dim=utter_dic_size+100,    #one-hotで表すことができる単語の数(10000+100で余裕を持たせる)
    output_dim=128,                   #1単語あたりのohe-hotの数：多くするほど単語の特徴を掴める(品詞をベクトルで表す)
    )(utterance)                    #データの前処理で形態要素を最大の数に合わせて行列で表した１つの品詞をベクトルで表す、形態要素の最大が50の場合は、50×128の行列になる　

# システムの応答: 入力は単語の総数+100、出力の次元数は128
emb_system = Embedding(
    input_dim=system_dic_size+100, # 応答の単語数+100
    output_dim=128                  # 出力の次元数はRecuurrent層のユニット数
    )(system)

#人間の発話したことばのトークン(品詞の分類数)をスカラーではなく、5次元で表す
emb_u_len = Embedding(
    input_dim=UTTER_MAX_SIZE+1,#入力の次元は発話の形態素数の最大値+1
    output_dim=5#出力は5
)(u_token_len)

# システムのシステムの発話したことばのトークン(品詞の分類数)をスカラーではなく、5次元で表す
emb_s_len = Embedding(
    input_dim=SYSTEM_MAX_SIZE+1,   # 入力の次元は応答の形態素数の最大値+1
    output_dim=5                   # 出力は5
    )(s_token_len)

#-----Reccurent層-----
#t=1～4までの時系列の場合、t=4の時系列データのみGRU層の3層目が全結合層に繋がる
#データの入力出力形状は最大のシーケンスの数×埋め込み次元の行列で表される
#Embedding層で各データはシーケンスの数が5次元ベクトルで表され、シーケンスの最大に対して、足りない行数は0で埋められる

#ユニット数128、すべての時刻の隠れ状態を出力　入力：埋め込み層（Embedding）の出力、[バッチサイズ、シーケンス、埋め込み次元]
run_layer1_1 = GRU(128,return_sequences=True)(emb_utterance)

#ユニット数128,すべての時刻の隠れ状態を出力　入力：前のそうrun_layer1_1の128ニューロンの隠れ状態、[バッチサイズ、シーケンス、埋め込み次元]
run_layer1_2 = GRU(128,return_sequences=True)(run_layer1_1)

#ユニット数128、シーケンスの最後の時刻の隠れ状態だけを出力　入力：前のそうrun_layer1_2の128ニューロンの隠れ状態、[バッチサイズ、シーケンス、埋め込み次元]
run_layer1_3 = GRU(128,return_sequences=False)(run_layer1_2)

#システムの応答
run_layer2_1 = GRU(128, return_sequences=True
                  )(emb_system)
run_layer2_2 = GRU(128, return_sequences=True
                  )(run_layer2_1)
run_layer2_3 = GRU(128, return_sequences=False
                  )(run_layer2_2)

#------全結合層------
main_l = concatenate([
    Flatten()(emb_u_len), # 人間の発話の単語数のEmbedding
    Flatten()(emb_s_len), # 商品名の単語数のEmbedding
    run_layer1_3,         # 人間の発話のGRUユニット
    run_layer2_3          # システム応答のGRUユニット
])

#-----512,256,128ユニット層の追加-----
main_l = Dropout(0.2)(
    Dense(512,kernel_initializer='normal',activation='relu')(main_l))
main_l = Dropout(0.2)(
    Dense(256,kernel_initializer='normal',activation='relu')(main_l))
main_l = Dropout(0.2)(
    Dense(128,kernel_initializer='normal',activation='relu')(main_l))

#-----出力層(3ユニット)----
output = Dense(units=3,              # 出力層のニューロン数＝2
                activation='softmax', # 活性化はソフトマックス関数
               )(main_l)

#Modelオブジェクトの生成
model = Model(
    # 入力層はマルチ入力モデルなのでリストにする
    inputs=[utterance, system,
            u_token_len, s_token_len
           ],
    # 出力層
    outputs=output
)

# Sequentialオブジェクをコンパイル
model.compile(
    loss='categorical_crossentropy', # 誤差関数はクロスエントロピー
    optimizer=optimizers.Adam(),     # Adamオプティマイザー
    metrics=['accuracy']             # 学習評価として正解率を指定
    )

model.summary()                      # RNNのサマリー（概要）を出力

'''
26.訓練データと正解ラベルの用意
'''
import numpy as np
from tensorflow.keras.utils import to_categorical

trainX = {
    # 人間の発話
    'utterance': train_U,
    # システムの応答
    'system': train_S,
    
    # 人間の発話の形態素の数(int)
    'u_token_len': np.array(df[['u_token_len']]),
    # システムの応答の形態素の数(int)
    's_token_len': np.array(df[['s_token_len']])
}

# 正解ラベルをOne-hot表現にする
trainY = to_categorical(df['label'], 3)

%%time
'''
27.学習の実行
'''
import math
from tensorflow.keras.callbacks import LearningRateScheduler

batch_size = 32            # ミニバッチのサイズ
lr_min = 0.0001            # 最小学習率
lr_max = 0.001             # 最大学習率

# 学習率をスケジューリングする
def step_decay(epoch):
    initial_lrate = 0.001 # 学習率の初期値
    drop = 0.25            # 減衰率は25%
    epochs_drop = 10.0    # 10エポック毎に減衰する
    lrate = initial_lrate * math.pow(
        drop,
        math.floor((epoch)/epochs_drop)
    )
    return lrate

# 学習率のコールバック
lrate = LearningRateScheduler(step_decay)

# エポック数
epoch = 40

# 学習を開始
history = model.fit(trainX, trainY,        # 訓練データ、正解ラベル
                    batch_size=batch_size, # ミニバッチのサイズ
                    epochs=epoch,          # 学習回数
                    verbose=1,             # 学習の進捗状況を出力する
                    validation_split=0.2,  # 訓練データの20%を検証データにする
                    shuffle=True,         # 検証データ抽出後にシャッフル
                    callbacks=[lrate]
                    )

