import BPLoaddataPart as BP
import BPTrainPart as TI
import time

if __name__ == "__main__":
    # 首先加载所需处理的数据
    load_data = BP.Load_Data()
    # 其次对频数数据进行预处理（Matlab之前给的数据有一定缺陷）
    load_data.pre_process_data()
    # 划分训练集和测试集，rate = 0.5，默认 为0.5
    load_data.divide_training_and_test_set(rate=0.4)

    # 下面是训练部分，首先把训练集输入到训练模块儿中
    # print(help(TI.Train_Data))
    print('下面开始训练啦！！！')
    # 以下是做图的部分
    axis_x = []
    axis_y = []
    fig1 = BP.plt.figure(1)
    axes = BP.plt.subplot(111)
    BP.plt.axis([0, 1000, 0, 1])
    axes.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    axes.grid(True)
    BP.plt.ylabel('Loss')
    BP.plt.xlabel('Epoch')
    # 作图部分结束，开始训练的部分
    train_data = TI.Train_Data(hiden_layers_num=2, neurons_num_per_layer=[12, 24, 12, 1],
                               learning_rate=0.03)
    for j in range(1000):
        for i in range(0, load_data.train_label.size):
            train_data.change_dataset(load_data.train_input[i], load_data.train_label[i])
            for count in range(0, 1):
                    output, caches = train_data.forward_propagation(mode='train')
                    grads = train_data.backward_propagation(caches)
                    train_data.update_w_and_b(grads)
                    loss, _ = train_data.compute_loss(caches[train_data.layers_num-1][0])
        # 输出精度
        print('当前迭代次数：', j)
        axis_x.append(j)
        train_data.change_dataset(load_data.test_input, load_data.test_label)
        output, caches = train_data.forward_propagation(mode='test')
        loss = train_data.compute_total_loss(caches[train_data.layers_num - 1][0])
        print(loss)
        axis_y.append(loss)
    # 输出精度结束
    # 下面是测试部分
    BP.plt.plot(axis_x, axis_y)
    print('下面开始测试啦！')
    train_data.change_dataset(load_data.test_input, load_data.test_label)
    output, caches = train_data.forward_propagation(mode='test')
    loss = train_data.compute_total_loss(caches[train_data.layers_num-1][0])
    print(loss)
    fig2 = BP.plt.figure(2)
    BP.plt.plot(axis_x, axis_y)
    BP.plt.show(fig1)
    time.sleep(5)
    BP.plt.close(1)
    BP.plt.show(fig2)
    time.sleep(5)
    BP.plt.close(2)