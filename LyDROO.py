#  #################################################################
#
#  This file contains the main code of LyDROO.
#
#  References:
#  [1] Suzhi Bi, Liang Huang, Hui Wang, and Ying-Jun Angela Zhang, "Lyapunov-guided Deep Reinforcement Learning for Stable Online Computation Offloading in Mobile-Edge Computing Networks," IEEE Transactions on Wireless Communications, 2021, doi:10.1109/TWC.2021.3085319.
#  [2] Liang Huang, Suzhi Bi, and Ying-Jun Angela Zhang, "Deep Reinforcement Learning for Online Offloading in Wireless Powered Mobile-Edge Computing Networks," in IEEE Transactions on Mobile Computing, vol. 19, no. 11, pp. 2581-2593, November 2020.
#  [3] S. Bi and Y. J. Zhang, “Computation rate maximization for wireless powered mobile-edge computing with binary computation ofﬂoading,” IEEE Trans. Wireless Commun., vol. 17, no. 6, pp. 4177-4190, Jun. 2018.
#
# version 1.0 -- July 2020. Written by Liang Huang (lianghuang AT zjut.edu.cn)
#  #################################################################


import scipy.io as sio                     # import scipy.io for .mat file I/
import numpy as np                         # import numpy

# Implementated based on the PyTorch
from memory import MemoryDNN
# import the resource allocation function
# replace it with your algorithm when applying LyDROO in other problems
from ResourceAllocation import Algo1_NUM

import math


def plot_rate( rate_his, rolling_intv = 50, ylabel='Normalized Computation Rate'):
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl

    rate_array = np.asarray(rate_his)
    df = pd.DataFrame(rate_his)


    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(15,8))

    plt.plot(np.arange(len(rate_array))+1, np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values), 'b')
    plt.fill_between(np.arange(len(rate_array))+1, np.hstack(df.rolling(rolling_intv, min_periods=1).min()[0].values), np.hstack(df.rolling(rolling_intv, min_periods=1).max()[0].values), color = 'b', alpha = 0.2)
    plt.ylabel(ylabel)
    plt.xlabel('Time Frames')
    plt.show()

# generate racian fading channel with power h and Line of sight ratio factor
# replace it with your own channel generations when necessary
def racian_mec(h,factor):
    n = len(h)
    beta = np.sqrt(h*factor) # LOS channel amplitude
    sigma = np.sqrt(h*(1-factor)/2) # scattering sdv
    x = np.multiply(sigma*np.ones((n)),np.random.randn(n)) + beta*np.ones((n))
    y = np.multiply(sigma*np.ones((n)),np.random.randn(n))
    g = np.power(x,2) +  np.power(y,2)
    return g


if __name__ == "__main__":
    '''
        LyDROO algorithm composed of four steps:
            1) 'Actor module'
            2) 'Critic module'
            3) 'Policy update module'
            4) ‘Queueing module’ of
    '''

    N =10                     # number of users
    n = 500                     # number of time frames
    K = N                   # initialize K = N
    decoder_mode = 'OPN'    # the quantization mode could be 'OP' (Order-preserving) or 'KNN' or 'OPN' (Order-Preserving with noise)
    Memory = 1024          # capacity of memory structure
    Delta = 32             # Update interval for adaptive K
    CHFACT = 10**10       # The factor for scaling channel value
    energy_thresh = np.ones((N))*0.08 # energy comsumption threshold in J per time slot
    nu = 1000 # energy queue factor;
    w = [1.5 if i%2==0 else 1 for i in range(N)] # weights for each user
    V = 20

    arrival_lambda = 3*np.ones((N)) # average data arrival, 3 Mbps per user

    print('#user = %d, #channel=%d, K=%d, decoder = %s, Memory = %d, Delta = %d'%(N,n,K,decoder_mode, Memory, Delta))

    # initialize data
    channel = np.zeros((n,N)) # chanel gains
    dataA = np.zeros((n,N))  # arrival data size


    # generate channel
    # 这行代码使用np.linspace()函数生成一个包含N个元素的数组，
    # 这些元素是从120到255的等间距的值，代表通道之间的距离。
    dist_v = np.linspace(start = 120, stop = 255, num = N)
    # Ad = 3: 这是一个常数，用于调整路径损耗模型中的某些参数。
    Ad = 3
    # fc = 915*10**6: 这是通信系统的载波频率，单位是赫兹。
    fc = 915*10**6
    # loss_exponent = 3: 这是路径损耗模型中的指数项，通常表示信号的衰减情况。
    loss_exponent = 3 # path loss exponent
    # light = 3*10**8: 这是光速，单位是米每秒。
    light = 3*10**8
    # h0 = np.ones((N)): 创建了一个长度为N的数组，初始值全部为1，用于存储通道增益。
    h0 = np.ones((N))
    # for j in range(0,N):: 这是一个循环，用于计算每个通道的增益。
    for j in range(0,N):
        # h0[j] = Ad*(light/4/math.pi/fc/dist_v[j])**(loss_exponent): 这行代码计算了第j个通道的增益。
        # 它使用了自由空间传播模型，这是一种用来描述无线信号在自由空间中传播时的衰减情况的模型
        h0[j] = Ad*(light/4/math.pi/fc/dist_v[j])**(loss_exponent)

    # 这是初始化了一个名为mem的MemoryDNN对象，它将用于存储和学习算法中的信息。
    mem = MemoryDNN(net = [N*3, 256, 128, N],
                    learning_rate = 0.01,
                    training_interval=20,
                    batch_size=128,
                    memory_size=Memory
                    )


    mode_his = [] # store the offloading mode
    k_idx_his = [] # store the index of optimal offloading actor
    Q = np.zeros((n,N)) # data queue in MbitsW
    Y = np.zeros((n,N)) # virtual energy queue in mJ
    Obj = np.zeros(n) # objective values after solving problem (26)
    energy = np.zeros((n,N)) # energy consumption
    rate = np.zeros((n,N)) # achieved computation rate


    # 这个部分包含了一个主要的循环，用于模拟LyDROO算法中的每个时间帧的操作。
    for i in range(n):
        # 这段代码每个10%的迭代输出一次当前完成的进度，以百分比的形式显示。
        # n//10用于获取迭代次数的十分之一，然后i % (n//10)用于检查当前迭代是否是十分之一的倍数，
        # 如果是，则打印当前进度。打印的内容是"%0.1f"%(i/n)，
        # 它将当前迭代次数除以总的迭代次数，然后以小数形式打印，保留一位小数。
        if i % (n//10) == 0:
            print("%0.1f"%(i/n))
        # 这部分代码用于更新LyDROO算法中的参数K，它控制着演员模块中可能选择的最大动作数。
        # Delta是更新间隔，它决定了何时更新K的值。如果当前迭代次数i大于0且可以被Delta整除，则进入更新。
        # 首先，代码检查Delta是否大于1，如果是，则从k_idx_his中获取最近Delta个元素，
        # 计算它们对K取模后的最大值加1，表示下一个可能的动作数。
        # 如果Delta等于1，则直接将上一个迭代的k_idx_his的最后一个元素加1作为max_k。
        # 然后，将max_k + 1 与N中的较小值作为新的K。
        if i> 0 and i % Delta == 0:
            # index counts from 0
            if Delta > 1:
                max_k = max(np.array(k_idx_his[-Delta:-1])%K) +1
            else:
                max_k = k_idx_his[-1] +1
            K = min(max_k +1, N)
        # 这行代码简单地将当前迭代次数i赋值给i_idx变量，用于后续的索引。
        i_idx = i

        #real-time channel generation
        # 这一行调用了名为racian_mec的函数，并传入了h0和0.3作为参数。
        # 根据函数名和参数的含义，它很可能是一个产生莱斯衰落增益的函数。
        # 返回的结果被赋值给h_tmp。
        h_tmp = racian_mec(h0,0.3)
        # increase h to close to 1 for better training; it is a trick widely adopted in deep learning
        # 这一行将通道增益h_tmp乘以CHFACT，以将通道增益调整到接近1的范围内。
        # 这个技巧是为了在训练深度学习模型时更好地处理数据，可能是因为一些算法对于较小的值更敏感或者更容易收敛。
        h = h_tmp*CHFACT
        # 这一行将计算得到的通道增益h赋值给channel数组的第i行，表示当前时间帧的通道增益。
        channel[i,:] = h
        # real-time arrival generation
        # 这一行生成实时到达数据量。它使用了指数分布来模拟数据到达的随机性，并且到达速率由参数arrival_lambda控制，
        # 该参数表示每个用户的平均到达速率。.random.exponential()函数返回一个满足指数分布的随机样本。
        dataA[i,:] = np.random.exponential(arrival_lambda)


        # 4) ‘Queueing module’ of LyDROO
        # 这段代码实现了LyDROO算法中的队列模块，用于更新数据队列和能量队列。
        # 检查当前时间索引i_idx是否大于0，以确保在第一个时间帧之后再执行队列更新操作。
        if i_idx > 0:
            # update queues
            # 这一行更新了数据队列。它通过从前一个时间帧的数据队列中减去上一个时间帧的计算速率（rate[i_idx-1, :]），
            # 并加上上一个时间帧的到达数据量（dataA[i_idx-1, :]）来计算当前时间帧的数据队列。
            Q[i_idx,:] = Q[i_idx-1,:] + dataA[i_idx-1,:] - rate[i_idx-1,:] # current data queue
            # assert Q is positive due to float error
            # 这一行通过检查数据队列中的负值并将其设置为0来确保数据队列是非负的。
            # 这个操作是由于浮点数运算可能导致数据队列中出现小的负值。
            Q[i_idx,Q[i_idx,:]<0] =0
            # 这一行更新了能量队列。
            # 它使用上一个时间帧的能量队列值（Y[i_idx-1, :]）加上能量消耗与能量阈值之间的差值
            # （energy[i_idx-1, :] - energy_thresh）乘以能量队列因子nu来计算当前时间帧的能量队列。
            Y[i_idx,:] = np.maximum(Y[i_idx-1,:] + (energy[i_idx-1,:]- energy_thresh)*nu,0) # current energy queue
            # assert Y is positive due to float error
            # 通过检查能量队列中的负值并将其设置为0来确保能量队列是非负的
            Y[i_idx,Y[i_idx,:]<0] =0

        # scale Q and Y to close to 1; a deep learning trick
        # 这一行代码将通道增益h、数据队列Q[i_idx, :]和能量队列Y[i_idx, :]连接成一个输入向量nn_input。
        # 之后的代码会使用这个向量作为模型的输入。
        nn_input =np.concatenate( (h, Q[i_idx,:]/10000,Y[i_idx,:]/10000))

        # 1) 'Actor module' of LyDROO
        # generate a batch of actions
        # 这行代码调用了mem对象的decode()方法，传入输入向量nn_input、动作数量K和解码器模式decoder_mode。
        # 它返回了一个动作列表m_list，这些动作是"Actor module"选择的。
        m_list = mem.decode(nn_input, K, decoder_mode)
        # 这两行初始化了两个空列表r_list和v_list，分别用于存储每个候选动作的结果和目标值。
        r_list = [] # all results of candidate offloading modes
        v_list = [] # the objective values of candidate offloading modes
        # 这是一个循环，遍历了所有候选动作列表m_list中的动作。
        for m in m_list:
            # 2) 'Critic module' of LyDROO
            # allocate resource for all generated offloading modes saved in m_list
            # 在每次循环中，调用Algo1_NUM函数来计算给定动作m的结果，并将结果添加到r_list列表中。
            # Algo1_NUM函数用于模拟资源分配或其他相关的计算，并返回一个包含结果的列表。
            r_list.append(Algo1_NUM(m,h,w,Q[i_idx,:],Y[i_idx,:],V))
            # 这一行将每个动作的目标值添加到v_list列表中。
            # 这里假设Algo1_NUM返回的结果是一个包含目标值的列表，
            # 因此[-1][0]用于获取列表中的第一个值（即目标值），并将其添加到v_list中。
            v_list.append(r_list[-1][0])
        # 这行代码记录了目标值列表v_list中最大值所对应的索引，并将该索引添加到k_idx_his列表中。
        # 这个索引表示了在当前时间步骤中选择的具有最大目标值的动作的索引。
        # record the index of largest reward
        k_idx_his.append(np.argmax(v_list))


        # 3) 'Policy update module' of LyDROO
        # encode the mode with largest reward
        # 这一行代码调用了 mem 对象的 encode() 方法。
        # 这个方法的目的是根据当前状态（nn_input）和选择的动作（m_list[k_idx_his[-1]]）来更新策略模型。
        # 具体来说，它会将这个状态和动作提供给策略模型，以便模型学习和调整。
        mem.encode(nn_input, m_list[k_idx_his[-1]])
        # 这行代码将选择的具有最大目标值的动作（m_list[k_idx_his[-1]]）添加到 mode_his 列表中。
        # 这个列表用于跟踪历史动作的选择记录。将选择的动作添加到 mode_his 中有助于后续分析和调试，以及对算法性能的评估。
        mode_his.append(m_list[k_idx_his[-1]])

        # 这一行代码将选择的具有最大目标值的动作所对应的结果从列表 r_list 中提取出来，并将其分配给对应的变量。
        # 具体来说：
        # Obj[i_idx]: 将选择动作的目标值存储在 Obj 数组的当前时间索引位置 i_idx 上。
        # rate[i_idx, :]: 将选择动作的计算速率存储在 rate 数组的当前时间索引位置 i_idx 上。
        # energy[i_idx, :]: 将选择动作的能量消耗存储在 energy 数组的当前时间索引位置 i_idx 上。

        # store max result
        Obj[i_idx],rate[i_idx,:],energy[i_idx,:]  = r_list[k_idx_his[-1]]


    mem.plot_cost()

    plot_rate(Q.sum(axis=1)/N, 100, 'Average Data Queue')
    plot_rate(energy.sum(axis=1)/N, 100, 'Average Energy Consumption')


    # save all data
    sio.savemat('./result_%d.mat'%N, {'input_h': channel/CHFACT,'data_arrival':dataA,'data_queue':Q,'energy_queue':Y,'off_mode':mode_his,'rate':rate,'energy_consumption':energy,'data_rate':rate,'objective':Obj})
