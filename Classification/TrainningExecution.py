import numpy as np
import pandas as pd
import InputData as datas
import Trainning as train

class TrainningExecution:
    def __init__(self, file_path):
        self.file_path = file_path

    def execute(self):
        # データの読み込み
        df = pd.read_csv(self.file_path)
        # データ加工
        data = datas.InputData(df)
        data.normalize()
        data.one_hot()
        data.separate_data_train_and_test()

        # 各設定値
        epoch = 100
        batch_size = 8
        wb_width = 0.1  # 重みとバイアスの広がり具合
        eta = 0.1  # 学習係数
        interval = 5  # 経過の表示間隔
        n_mid = 25  # 中間層のニューロン数

        tr = train.Trainning(data)
        tr.delete()
        tr.prepare(epoch, batch_size, eta, wb_width, interval, n_mid)
        tr.layers_init()
        # 学習と誤差の記録
        tr.execute()
        # 正解率の計算
        tr.calc＿accuracy＿rate()
        # 結果を保存
        tr.save(self.file_path)

        # 学習係数を変えて再実行
        tr = train.Trainning(data)
        eta = 0.01
        tr.prepare(epoch, batch_size, eta, wb_width, interval, n_mid)
        tr.layers_init()
        # 学習と誤差の記録
        tr.execute()
        # 正解率の計算
        tr.calc＿accuracy＿rate()
        # 結果を保存
        tr.save(self.file_path)

