#landmark_nn.py
%matplotlib inline
import matplotlib.pyplot as plt
class MyNN():
  
  def __init__(self, optimizer, net, n_epoch, n_batchsize):
    self.optimizer_ = optimizer
    self.net_ = net
    self.optimizer_.setup(self.net_)
    self.n_epoch_ = n_epoch
    self.n_batchsize_ = n_batchsize
    self.max_accuracy_train = 0
    self.max_accuracy_valid = 0
    # ログの保存用
    self.results_train_ = {
      'loss': [],
      'accuracy': []
    }
    self.results_valid_ = {
      'loss': [],
      'accuracy': []
    }

  def printNet(self):
    print(self.net_)
  def fit(self, x_train, t_train, x_val, y_val):
    iteration = 0

    for epoch in range(self.n_epoch_):

      # データセット並べ替えた順番を取得
      order = np.random.RandomState(seed=20190425).permutation(range(len(x_train)))

      # 各バッチ毎の目的関数の出力と分類精度の保存用
      loss_list = []
      accuracy_list = []

      for i in range(0, len(order), self.n_batchsize_):
        # バッチを準備
        index = order[i:i+self.n_batchsize_]
        x_train_batch = x_train[index,:]
        t_train_batch = t_train[index]

        # 予測値を出力
        y_train_batch = self.net_(x_train_batch)

        # 目的関数を適用し、分類精度を計算
        loss_train_batch = F.softmax_cross_entropy(y_train_batch, t_train_batch)
        accuracy_train_batch = F.accuracy(y_train_batch, t_train_batch)

        loss_list.append(loss_train_batch.array)
        accuracy_list.append(accuracy_train_batch.array)

        # 勾配のリセットと勾配の計算
        self.net_.cleargrads()
        loss_train_batch.backward()

        # パラメータの更新
        self.optimizer_.update()

        # カウントアップ
        iteration += 1

      # 訓練データに対する目的関数の出力と分類精度を集計
      loss_train = np.mean(loss_list)
      accuracy_train = np.mean(accuracy_list)

      # 1エポック終えたら、検証データで評価
      # 検証データで予測値を出力
      with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        y_val = self.net_(x_val)

      # 目的関数を適用し、分類精度を計算
      loss_val = F.softmax_cross_entropy(y_val, t_val)
      accuracy_val = F.accuracy(y_val, t_val)

      # 結果の表示
      #print('epoch: {}, iteration: {}, loss (train): {:.4f}, loss (valid): {:.4f}'.format(
      #    epoch, iteration, loss_train, loss_val.array))

      # ログを保存
      self.results_train_['loss'] .append(loss_train)
      self.results_train_['accuracy'] .append(accuracy_train)
      self.results_valid_['loss'].append(loss_val.array)
      self.results_valid_['accuracy'].append(accuracy_val.array)
      self.max_accuracy_train = max(accuracy_train,self.max_accuracy_train)
      self.max_accuracy_valid = max(accuracy_val.array,self.max_accuracy_valid)

    
  def fit_info(self, loss_or_accuracy):
    plt.clf()
    plt.plot(self.results_train_[loss_or_accuracy], label='train')  # label で凡例の設定
    plt.plot(self.results_valid_[loss_or_accuracy], label='valid') 
    plt.legend()  # 凡例の表示
    plt.xlabel('epoch')
    plt.ylabel(loss_or_accuracy)
    plt.show()
    print(self.max_accuracy_train)
    print(self.max_accuracy_valid)
  def predict(self,x):
    # テストデータで予測値を計算
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
      s = self.net_(x)
    y = [np.argmax(s[i,:].array) for i in range(len(s))]
    return y
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Sequential
class Net():
	def __init__(self):
		chainer.print_runtime_info()
		# net としてインスタンス化
		self.n_input = 7
		self.n_hidden_1 = 5
		self.n_hidden_2 = 5
		self.n_output = 2

		self.net = Sequential(
		    L.Linear(n_input, n_hidden_1), F.relu,
		    L.Linear(n_hidden_1, n_hidden_1), F.relu,
		    L.Linear(n_hidden_1, n_hidden_1), F.relu,
		    L.Linear(n_hidden_1, n_hidden_1), F.relu,
		    L.Linear(n_hidden_1, n_hidden_1), F.relu,
		    
		    L.Linear(n_hidden_1, n_hidden_1), F.relu,
		    L.Linear(n_hidden_1, n_hidden_2), F.relu,
		    L.Linear(n_hidden_2, n_hidden_2), F.relu,
		    L.Linear(n_hidden_2, n_hidden_2), F.relu,
		    L.Linear(n_hidden_2, n_hidden_2), F.relu,
		    L.Linear(n_hidden_2, n_hidden_2), F.relu,
		    L.Linear(n_hidden_2, n_output)
		)