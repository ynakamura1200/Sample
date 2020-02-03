# coding: utf-8
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Date, DECIMAL
from assets.database import Base
from datetime import datetime as dt

class Conditions(Base):
    __tablename__ = "CONDITIONS"
    __table_args__ = {'extend_existing': True}
    id = Column(Integer, primary_key=True, autoincrement=True)
    file_path = Column(String(200), unique=False)
    total_epoch = Column(Integer, unique=False)
    batch_size = Column(Integer, unique=False)
    eta = Column(DECIMAL(10, 7), unique=False)
    wb_width = Column(Integer, unique=False)
    n_in = Column(Integer, unique=False)
    n_mid = Column(Integer, unique=False)
    n_out = Column(Integer, unique=False)
    updated_at = Column(DateTime, default=dt.now())
    created_at = Column(DateTime, default=dt.now())

    def __init__(self, file_path=None, total_epoch=None, batch_size=None, eta=None, wb_width=None, n_in=None, n_mid=None, n_out=None, updated_at=None, created_at=None):
        self.file_path = file_path
        self.total_epoch = total_epoch
        self.batch_size = batch_size
        self.eta = eta
        self.wb_width = wb_width
        self.n_in = n_in
        self.n_mid = n_mid
        self.n_out = n_out
        self.updated_at = updated_at
        self.created_at = created_at

class Loss(Base):
    __tablename__ = "LOSS"
    __table_args__ = {'extend_existing': True}
    id = Column(Integer, primary_key=True, autoincrement=True)
    condition_id = Column(Integer, unique=False)
    epoch = Column(Integer, unique=False)
    train_loss = Column(DECIMAL(10, 7), unique=False)
    test_loss = Column(DECIMAL(10, 7), unique=False)
    updated_at = Column(DateTime, default=dt.now())
    created_at = Column(DateTime, default=dt.now())

    def __init__(self, condition_id=None, epoch=None, train_loss=None, test_loss=None, updated_at=None, created_at=None):
        self.condition_id = condition_id
        self.epoch = epoch
        self.train_loss = train_loss
        self.test_loss = test_loss
        self.updated_at = updated_at
        self.created_at = created_at

class Accuracy(Base):
    __tablename__ = "ACCUARACY"
    __table_args__ = {'extend_existing': True}
    condition_id = Column(Integer, primary_key=True)
    train_accuracy = Column(DECIMAL(10, 7), unique=False)
    test_accuracy = Column(DECIMAL(10, 7), unique=False)
    updated_at = Column(DateTime, default=dt.now())
    created_at = Column(DateTime, default=dt.now())

    def __init__(self, condition_id=None, train_accuracy=None, test_accuracy=None, updated_at=None, created_at=None):
        self.condition_id = condition_id
        self.train_accuracy = train_accuracy
        self.test_accuracy = test_accuracy
        self.updated_at = updated_at
        self.created_at = created_at
