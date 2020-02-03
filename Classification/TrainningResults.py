class TrainningResults:
    def __init__(self):
        # -- 誤差の記録用 --
        self.epoch_losses = []
        self.accuracy_train = 0
        self.accuracy_test = 0

    def record_loss(self, epoch, train_loss, test_loss):
       self.epoch_losses.append([epoch, train_loss, test_loss])

    def calc_train_accuracy(self, count_train, n_train):
        self.accuracy_train = count_train/n_train*100

    def calc_test_accuracy(self, count_test, n_test):
        self.accuracy_test = count_test/n_test*100

    def calc_accuracy(self, count_train, n_train, count_test, n_test):
        self.calc_train_accuracy(count_train, n_train)
        self.calc_test_accuracy(count_test, n_test)
