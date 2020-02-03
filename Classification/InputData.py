import numpy as np
import pandas as pd

class InputData:
    def __init__(self, df):
        self.input_data = df.iloc[:, 1:].values
        self.input_n_cloumn = len(df.columns) -1
        self.correct = df.iloc[:, 0].values
        self.n_correct = len(set(self.correct))
        self.correct_data = []
        self.n_data = len(self.correct) #サンプル数
        self.index_train = []
        self.index_test = []
        self.input_train = []
        self.input_test = []
        self.correct_train = []
        self.correct_test = []
        self.n_train = 0
        self.n_test = 0
        
    def normalize(self):
        # 入力データを標準化する
        ave_input = np.average(self.input_data, axis=0)
        std_input = np.std(self.input_data, axis=0)
        self.input_data = (self.input_data - ave_input) / std_input

    def one_hot(self):
        # 正解をone-hot表現にする
        self.correct_data = np.zeros((self.n_data, self.n_correct))
        for i in range(self.n_data):
            self.correct_data[i, self.correct[i]] = 1.0
        
    def separate_data_train_and_test(self):
        # 訓練データとテストデータに分ける
        index = np.arange(self.n_data)
        self.index_train = index[index%2 == 0]
        self.index_test = index[index%2 != 0]

        self.input_train = self.input_data[self.index_train, :]  # 訓練 入力
        self.correct_train = self.correct_data[self.index_train, :]  # 訓練 正解
        self.input_test = self.input_data[self.index_test, :]  # テスト 入力
        self.correct_test = self.correct_data[self.index_test, :]  # テスト 正解

        self.n_train = self.input_train.shape[0]  # 訓練データのサンプル数
        self.n_test = self.input_test.shape[0]  # テストデータのサンプル数
