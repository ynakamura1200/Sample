
import datetime
import numpy as np
import pandas as pd

from assets import models
from assets.database import db_session
from Layers import MiddleLayer
from Layers import OutputLayer
import TrainningResults as tres

class Trainning:
    res = None
    middle_layer_1 = None
    middle_layer_2 = None
    output_layer = None

    def __init__(self, data):
        self.data = data # InputData class
        self.epoch = 0
        self.batch_size = 0
        self.eta = 0 # 学習係数
        self.wb_width = 0 # 重みとバイアスの広がり具合
        self.interval = 0 # 経過の表示間隔
        self.n_in = 0 # 入力層のニューロン数
        self.n_mid = 0 # 中間層のニューロン数
        self.n_out = 0 # 出力層のニューロン数

    def prepare(self, epoch, batch_size, eta, wb_width, interval, n_mid):
        self.res = tres.TrainningResults()
        self.epoch = epoch
        self.batch_size = batch_size
        self.eta = eta # 学習係数
        self.wb_width = wb_width # 重みとバイアスの広がり具合
        self.interval = interval # 経過の表示間隔
        self.n_in = self.data.input_n_cloumn # 入力層のニューロン数
        self.n_mid = n_mid # 中間層のニューロン数
        self.n_out = self.data.n_correct # 出力層のニューロン数
        
    def layers_init(self):
        # -- 各層の初期化 --
        self.middle_layer_1 = MiddleLayer(self.wb_width, self.n_in, self.n_mid)
        self.middle_layer_2 = MiddleLayer(self.wb_width, self.n_mid, self.n_mid)
        self.output_layer = OutputLayer(self.wb_width, self.n_mid, self.n_out)

    def execute(self):
        data = self.data

        # 学習と経過の記録
        n_batch = data.n_train // self.batch_size  # 1エポックあたりのバッチ数
        for i in range(self.epoch):

            # 誤差の計測
            self.forward_propagation(data.input_train)
            loss_train = self.get_loss(data.correct_train, data.n_train)
            self.forward_propagation(data.input_test)
            loss_test = self.get_loss(data.correct_test, data.n_test)
    
            # 誤差の記録 
            self.res.record_loss(i, loss_train, loss_test)
    
            # 経過の表示 
            if i%self.interval == 0:
                print("Epoch:" + str(i) + "/" + str(self.epoch),
                    "loss_train:" + str(loss_train),
                    "loss_test:" + str(loss_test))

            # 学習
            index_random = np.arange(data.n_train)
            np.random.shuffle(index_random)  # インデックスをシャッフルすxる
            for j in range(n_batch):
        
                # ミニバッチを取り出す
                mb_index = index_random[j*self.batch_size : (j+1)*self.batch_size]
                x = data.input_train[mb_index, :]
                t = data.correct_train[mb_index, :]
        
                # 順伝播と逆伝播
                self.forward_propagation(x)
                self.back_propagation(t)
        
                # 重みとバイアスの更新
                self.uppdate_wb()
        
        # -- 順伝播 --
    def forward_propagation(self, x):
        self.middle_layer_1.forward(x)
        self.middle_layer_2.forward(self.middle_layer_1.y)
        self.output_layer.forward(self.middle_layer_2.y)

        # -- 逆伝播 --
    def back_propagation(self, t):
        self.output_layer.backward(t)
        self.middle_layer_2.backward(self.output_layer.grad_x)
        self.middle_layer_1.backward(self.middle_layer_2.grad_x)

    # -- 重みとバイアスの更新 --
    def uppdate_wb(self):
        self.middle_layer_1.update(self.eta)
        self.middle_layer_2.update(self.eta)
        self.output_layer.update(self.eta)

    # -- 誤差を計算 --
    def get_loss(self, t, batch_size):
        return -np.sum(t * np.log(self.output_layer.y + 1e-7)) / self.batch_size  # 交差エントロピー誤差
    
    # 正解率の計算 
    def calc＿accuracy＿rate(self):
        data = self.data
        self.forward_propagation(data.input_train)
        count_train = np.sum(np.argmax(self.output_layer.y, axis=1) == np.argmax(data.correct_train, axis=1))
        self.forward_propagation(data.input_test)
        count_test = np.sum(np.argmax(self.output_layer.y, axis=1) == np.argmax(data.correct_test, axis=1))

        self.res.calc_accuracy(count_train, data.n_train, count_test, data.n_test)

    def save(self, file_path):
        date = datetime.datetime.today()
        condition = models.Conditions(file_path=file_path, total_epoch=self.epoch, \
            batch_size=self.batch_size, eta=self.eta, wb_width=self.wb_width, \
            n_in=self.n_in, n_mid=self.n_mid, n_out=self.n_out, updated_at=date, created_at=date)
        response = db_session.execute(models.Conditions.__table__.insert(), vars(condition))
        incremented_key = response.inserted_primary_key[0]
        res = self.res
        for ep_loss in res.epoch_losses:
            res.epoch_losses
            loss = models.Loss(condition_id=incremented_key, epoch=ep_loss[0], \
                train_loss=ep_loss[1], test_loss=ep_loss[2], updated_at=date, created_at=date)
            db_session.add(loss)
        
        accuracy = models.Accuracy(condition_id=incremented_key, train_accuracy=res.accuracy_train, \
            test_accuracy=res.accuracy_test, updated_at=date, created_at=date)
        db_session.add(accuracy)
        db_session.commit()

    def get_result(self):
        return self.res

    def delete(self):
        db_session.query(models.Conditions).delete()
        db_session.query(models.Loss).delete()
        db_session.query(models.Accuracy).delete()
        db_session.commit()
