#  #################################################################
#  This file contains the main DROO operations, including building DNN, 
#  Storing data sample, Training DNN, and generating quantized binary offloading decisions.

#  version 1.0 -- February 2020. Written based on Tensorflow 2 by Weijian Pan and 
#  Liang Huang (lianghuang AT zjut.edu.cn)
#  ###################################################################

from __future__ import print_function
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np



# DNN network for memory
class MemoryDNN:
    def __init__(
        self,
        net,
        learning_rate = 0.01,
        training_interval=10,
        batch_size=100,
        memory_size=1000,
        output_graph=False
    ):

        self.net = net
        self.training_interval = training_interval      # learn every #training_interval
        self.lr = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size

        # store all binary actions
        self.enumerate_actions = []

        # stored # memory entry
        self.memory_counter = 1

        # store training cost
        self.cost_his = []

        # initialize zero memory [h, m]
        self.memory = np.zeros((self.memory_size, self.net[0] + self.net[-1]))

        # construct memory network
        self._build_net()

    def _build_net(self):
        self.model = nn.Sequential(
                nn.Linear(self.net[0], self.net[1]),
                nn.ReLU(),
                nn.Linear(self.net[1], self.net[2]),
                nn.ReLU(),
                nn.Linear(self.net[2], self.net[3]),
                nn.Sigmoid()
        )

    def remember(self, h, m):
        # replace the old memory with new memory
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = np.hstack((h, m))

        self.memory_counter += 1

    def encode(self, h, m):
        # encoding the entry
        self.remember(h, m)
        # train the DNN every multiple steps

        if self.memory_counter % self.training_interval == 0:
            self.learn()

    def learn(self):
        # sample batch memory from all memory
        # 这一部分代码根据记忆库中是否存满了数据，从所有记忆中随机抽样一批数据。
        # 如果记忆库已满，则从记忆库的大小范围内随机抽取数据；否则，从记忆库当前大小的范围内随机抽取数据。
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        # 这一行将从记忆库中抽样得到的数据索引应用到记忆库中，以获取相应的批次数据。
        # 这样就得到了训练神经网络模型所需的输入-输出数据。
        batch_memory = self.memory[sample_index, :]
        # print('batch_memory',batch_memory)
        # 这两行代码从批次数据中分别提取出输入状态和输出动作，
        # 并将它们转换为 PyTorch 的 Tensor 格式，以便后续在神经网络中使用。
        # 在深度学习中，神经网络处理的输入数据通常是张量（tensor），
        # 其形状（shape）通常为 (batch_size, input_size)，
        # 其中 batch_size 表示批次大小，input_size 表示每个样本的输入特征数量。
        h_train = torch.Tensor(batch_memory[:, 0: self.net[0]])
        m_train = torch.Tensor(batch_memory[:, self.net[0]:])


        # train the DNN
        # 这一行代码定义了优化器，这里使用的是 Adam 优化器。
        # Adam 是一种常用的梯度下降算法，用于更新神经网络模型的参数。
        # self.model.parameters() 表示优化器要更新的参数是神经网络模型中的所有可训练参数，
        # lr=self.lr 指定了学习率，betas=(0.09, 0.999) 是 Adam 优化器的超参数，
        # weight_decay=0.0001 是 L2 正则化项的系数，用于控制模型的复杂度。
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr,betas = (0.09,0.999),weight_decay=0.0001)
        # 这一行代码定义了损失函数，这里使用的是二元交叉熵损失函数（Binary Cross EntropyLoss）。
        # 该损失函数通常用于二分类问题，用于计算模型预测值和真实标签之间的差异。
        criterion = nn.BCELoss()
        # 这一行代码将神经网络模型设置为训练模式。
        # 在训练模式下，模型的行为会有所不同，例如在反向传播过程中会计算梯度并更新参数。
        self.model.train()
        # 这一行代码用于清除之前的梯度，以确保在进行当前批次的梯度计算之前不会受到之前批次的影响。
        optimizer.zero_grad()
        # 这一行代码将输入数据 h_train 输入到神经网络模型中，以获取模型对输入数据的预测结果。
        predict = self.model(h_train)
        # 这一行代码计算了模型预测值 predict 和真实标签 m_train 之间的损失值，
        # 通过比较模型的预测输出和实际标签的差异来衡量模型的性能。
        loss = criterion(predict, m_train)
        # 这一行代码执行了反向传播过程，用于计算损失函数关于模型参数的梯度。
        loss.backward()
        # 这一行代码根据梯度更新模型的参数，以最小化损失函数。
        # 这是优化器的核心步骤，通过梯度下降法来调整模型参数，使损失函数的值尽可能小。
        optimizer.step()
        # 这一行代码将损失值保存在 self.cost 变量中，以便后续跟踪和记录。
        self.cost = loss.item()
        # 这一行代码用于断言损失值大于 0，以确保损失值有效且合理。
        assert(self.cost > 0)
        # 这一行代码将损失值添加到损失历史记录列表 self.cost_his 中，以便后续绘制损失曲线。
        self.cost_his.append(self.cost)

    def decode(self, h, k = 1, mode = 'OP'):
        # to have batch dimension when feed into Tensor
        # 这一行将输入的状态 h 转换成 PyTorch 的 Tensor，并增加了一个额外的维度，
        # 以符合神经网络的输入要求。这样做是因为神经网络模型接受的输入通常是一个批次的数据，即包含一个批次的状态数据。
        h = torch.Tensor(h[np.newaxis, :])
        # 这一行将神经网络模型切换到评估模式。在评估模式下，模型的行为会略有不同，例如不会进行梯度计算和参数更新。
        self.model.eval()
        # 这一行将输入状态 h 输入到神经网络模型中，以获取对应的输出预测。
        # 这里假设神经网络模型已经训练好，并且通过 self.model 访问。
        m_pred = self.model(h)
        # 这一行将预测结果从PyTorchTensor格式转换为NumPy数组，以便后续处理。
        m_pred = m_pred.detach().numpy()


        if mode == 'OP':
            return self.knm(m_pred[0], k)
        elif mode == 'KNN':
            return self.knn(m_pred[0], k)
        elif mode == 'OPN':
            return self.opn(m_pred[0], k)
        else:
            print("The action selection must be 'OP' or 'KNN' or 'OPN'")

    def knm(self, m, k = 1):
        # return k order-preserving binary actions
        m_list = []
        # generate the ﬁrst binary ofﬂoading decision with respect to equation (8)
        m_list.append(1*(m>0.5))

        if k > 1:
            # generate the remaining K-1 binary ofﬂoading decisions with respect to equation (9)
            m_abs = abs(m-0.5)
            idx_list = np.argsort(m_abs)[:k-1]
            for i in range(k-1):
                if m[idx_list[i]] >0.5:
                    # set the \hat{x}_{t,(k-1)} to 0
                    m_list.append(1*(m - m[idx_list[i]] > 0))
                else:
                    # set the \hat{x}_{t,(k-1)} to 1
                    m_list.append(1*(m - m[idx_list[i]] >= 0))

        return m_list
    
    def opn(self, m, k= 1):
        return self.knm(m,k)+self.knm(m+np.random.normal(0,1,len(m)),k)

    def knn(self, m, k = 1):
        # list all 2^N binary offloading actions
        if len(self.enumerate_actions) == 0:
            import itertools
            self.enumerate_actions = np.array(list(map(list, itertools.product([0, 1], repeat=self.net[0]))))

        # the 2-norm
        sqd = ((self.enumerate_actions - m)**2).sum(1)
        idx = np.argsort(sqd)
        return self.enumerate_actions[idx[:k]]


    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his))*self.training_interval, self.cost_his)
        plt.ylabel('Training Loss')
        plt.xlabel('Time Frames')
        plt.show()

