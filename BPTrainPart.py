from BPActivationPart import *

class Train_Data:
    def __init__(self, hiden_layers_num=1, neurons_num_per_layer=[12, 48, 1],
                 learning_rate=0.314):
        """"
        输入层数时，只用输入隐藏神经元的个数
        每层神经元的个数：是包括输入层和输出层的
        根据输入的信息配置好每一层的权重w，偏差b
        self.param_layers_num: 实际需要配备w和b的层数 3层
        """
        self.layers_num = hiden_layers_num + 2
        self.neurons_num_per_layer = neurons_num_per_layer
        self.learning_rate = learning_rate
        self.w = [np.random.normal(0.0, pow(self.neurons_num_per_layer[i-1], -0.5), (self.neurons_num_per_layer[i],
                                              self.neurons_num_per_layer[i-1])) for i in range(1,self.layers_num)]
        self.b = [np.zeros((self.neurons_num_per_layer[i], 1)) for i in range(1, self.layers_num)]

    def change_dataset(self, train_data, train_label):
        self.train_data = train_data.T / 10000.0 + 0.01
        self.train_label = train_label

    def layer_forward_propagation(self, activation_choose, layer_index, input, mode):
        """
        函数：每一层的正向传播
        输入：输入到该层中的信息 1. activation_choose(激活函数) 2. layer_index（输入层不算）(哪一层0,1,2) 3.input 输入该层的数据
        功能：进行 1.权重相乘 2.加上偏差 （前两个完成以后，会给名称加上sum）3.激活函数激活
        输出：经过1.output,激活两层处理后的信息;2.小括号内的，回去要调整：输入到函数中的信息
        """
        input = np.squeeze(input)
        if mode == 'train':
            input_sum = np.dot(self.w[layer_index], input) + np.squeeze(self.b[layer_index])
        elif mode == 'test':
            a = np.array(input.shape[1] * [self.b[layer_index]])
            input_sum = np.dot(self.w[layer_index], input) + a.T
        output, _ = activiated(activation_choose, input_sum)
        return output, input, input_sum, layer_index, activation_choose

    def forward_propagation(self, mode='train'):
        """
         函数：向前传播总函数
         功能：搭建正向传播总架构
         输入：caches(为存储架构每一层重要信息的空间)
         输出：1.output:最终输出 2. caches：储存的网络信息
        """
        caches = []
        # 对于输入层而言，输入就等于输出：
        cache = (self.train_data, self.train_data, self.train_data, -1, 'None')
        caches.append(cache)
        # 对于隐藏层采取的策略不同
        for i in range(1, self.layers_num - 1):
                cache = self.layer_forward_propagation('leakyrelu', i-1, caches[i-1][0], mode)
                caches.append(cache)
        # 解决最后一层，输出层！
        cache = self.layer_forward_propagation('sigmoid', self.layers_num - 2, caches[self.layers_num - 2][0], mode)
        caches.append(cache)
        return np.squeeze(cache[0]), caches

    def compute_loss(self, output):
        """
        函数: 计算误差
        输入：需要1.期望的输出值 2. 实际网络给出的输出值: output
        功能：计算上面所说的期望的输出值和实际网络给出的输出值之间的差值（这里采用交叉熵的方式）
        输出：我们改成最传统的方式
        """
        loss = np.power((self.train_label - output), 2)
        loss = np.squeeze(loss)
        dloss_nd_doutput = 2 * (output - self.train_label)
        return loss, dloss_nd_doutput

    def compute_total_loss(self, output):
        """
         函数: 计算总体的误差
         输入：需要1.期望的输出值 2. 实际网络给出的输出值: output
         功能：计算上面所说的期望的输出值和实际网络给出的输出值究竟能识别到哪种程度
         输出：我们改成最传统的方式
       """
        size = np.size(output)
        output = np.squeeze(output)
        count = 0
        for i in range(size):
            if output[i] > 0.5:
                output[i] = 0.99
            else:
                output[i] = 0.01
            if output[i] == self.train_label[i]:
                count += 1

        loss = count / size
        return loss

    def layer_backward_propagation(self, cache, dloss_nd_doutput):
        """
        函数：每一层的反向传播函数
        输入：1.cache:该层存储的必要信息 2. dloss_nd_doutput: 求误差对总输出的微分
        功能：进行每一层对应的权重更新大小的计算，输入层没有权重更新这一说法
        输出：1.dloss_nd_dw: 对权重w的更新信息 2. dloss_nd_db：对偏差b的更新信息 3. dloss_nd_dinput: 微分反向传递到上一层
        """
        input = cache[1]
        input_sum = cache[2]
        dloss_nd_dinput_sum = activated_back_propagation(cache[4], dloss_nd_doutput, input_sum)  # 目前大小为 1* 25000

        m = np.size(cache[0])
        if dloss_nd_dinput_sum.size == 1:
            dloss_nd_dw = dloss_nd_dinput_sum * input.T / m
        else:
            c = dloss_nd_dinput_sum.reshape((dloss_nd_dinput_sum.size, 1))
            e = input.reshape((1, input.size))
            dloss_nd_dw = np.dot(c, e)/m
        # sum函数的原型为sum(a, axis=0)，
        # 在默认状况下axis=0，意思是普通的相加，当axis=1时，
        # 为每行向量相加，但是数组要大于一维，否则会报错。
        dloss_nd_db = dloss_nd_dinput_sum / m
        dloss_nd_dinput = np.dot(self.w[cache[3]].T, dloss_nd_dinput_sum)
        return dloss_nd_dinput, dloss_nd_dw, dloss_nd_db

    def backward_propagation(self, caches):
        """
        函数：反向传播总函数
        输入：1.caches:正向传播的时候存储的所有层的信息。
        功能：搭建反向传播总架构
        输出：grad,包含更新权重的梯度信息
        """
        grad = {}
        output = caches[self.layers_num - 1][0]
        _, dloss_nd_doutput = self.compute_loss(output)
        for i in reversed(range(1, self.layers_num)):
            grad['dinput' + str(i)], grad['dw' + str(i)], grad['db' + str(i)] = self.layer_backward_propagation(caches[i], dloss_nd_doutput)
            dloss_nd_doutput = grad['dinput' + str(i)]
        return grad

    def update_w_and_b(self, grad):
        """
        函数：更新 w和b的权重值
        输入：1.grad:包含更新权重和偏差的梯度信息
        功能：更新权重和偏差
        输出：更新后的权重值和偏差值
        """
        for i in range(1, self.layers_num):
            self.w[i - 1] = self.w[i - 1] - self.learning_rate * grad['dw' + str(i)]
            self.b[i - 1] = np.squeeze(self.b[i - 1]) - np.squeeze(self.learning_rate * grad['db' + str(i)])



