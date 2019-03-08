# coding=gbk
import h5py
import sklearn.datasets
import sklearn.linear_model
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)

font = fm.FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')# ��һ����Ϊ��������������
matplotlib.rcParams['savefig.dpi'] = 300# ����ͼƬ����
matplotlib.rcParams['figure.figsize'] = (3,2)

def sigmoid(input_sum):
    """
    ������
            �����Sigmoid
    ���룺
            param input_sum: ���룬����Ԫ�ļ�Ȩ��
    ���أ�
            return:���������
            input_sum: �����뻺����������
    """
    output = 1.0/(1+np.exp(-input_sum))
    return output, input_sum

def sigmoid_back_propagation(derror_wrt_output, input_sum):
    """
    ������
            ��������Ԫ�����ƫ��: dE/dIn = dE/dOut * dOut/dIn  PDF��ʽ5.6
            ���У� dOut/dIn   ���Ǽ�����ĵ���  dy=y(1-y),PDF��ʽ��5.9��
    ���룺
         derror_wrt_output����������Ԫ�����ƫ������ʽ��5.7��
         input_sum: �����Ȩ��
    ���أ�
        derror_wrt_dinputs: �����������ƫ������ʽ��5.13��
    """
    output = 1.0/(1 + np.exp(- input_sum))
    doutput_wrt_dinput = output * (1 - output)
    derror_wrt_dinput = derror_wrt_output * doutput_wrt_dinput

    return derror_wrt_dinput

def relu(input_sum):
    """
    ������
            �����ReLU
    ���룺
            input_sum:���룬����Ԫ�ļ�Ȩ��
    ���أ�
            outputs:���������
            input_sum�������뻺����������
    """
    output = np.maximum(0, input_sum)
    return output, input_sum

def relu_back_propagation(derror_wrt_output, input_sum):
    """
    ������
            �����������������ƫ����dE/dIn = dE/dOut * dOut/dIn
            ���У�dOut/dIn ���Ǽ�����ĵ���
                  dE/dOut ������Ԫ�����ƫ��
    ���룺
            derror_wrt_output:��������Ԫ�����ƫ��
            input_sum: �����Ȩ��
    ���أ�
            derror_wrt_dinputs: �����������ƫ��
     ??? �����check һ�µ����Ǹ���ʽ�ܲ�����
    """
    derror_wrt_dinputs = np.array(derror_wrt_output, copy=True)
    #derror_wrt_dinputs[input_sum <= 0] = 0
    derror_wrt_dinputs = np.where(derror_wrt_dinputs > 0.0, derror_wrt_dinputs, 0)
    return derror_wrt_dinputs

def tanh(input_sum):
    """
    ������
            ����� tanh
    ���룺
            input_sum:���룬����Ԫ�ļ�Ȩ��
    ���أ�
            output: ���������
            input_sum�������뻺����������
    """
    output = np.tanh(input_sum)
    return output, input_sum

def tanh_back_propagation(derror_wrt_output, input_sum):
    """
    ������
            ��������Ԫ�����ƫ����dE/dIn = dE/dOut * dOut/dIn
            ���У�dOut/dIn ���Ǽ�����ĵ��� tanh'(x) = 1- x^2
                  dE/dOut ������Ԫ�����ƫ��
    ���룺
            derror_wrt_output:��������Ԫ�����ƫ��:��ʽ��5.7��
            input_sum: �����Ȩ��
    ���أ�
            derror_wrt_dinputs:�����������ƫ��
    """
    output = np.tanh(input_sum)
    doutput_wrt_dinput = 1 - np.power(output,2)
    derror_wrt_dinput = derror_wrt_output * doutput_wrt_dinput

    return derror_wrt_dinput

def leakyrelu(input_sum, Alpha=0.25):
    """
        ������
                ����� Leaky ReLU
        ���룺
                input_sum:���룬����Ԫ�ļ�Ȩ��
        ���أ�
                output: ���������
                input_sum�������뻺����������
    """
    # output = np.squeeze([i * Alpha for i in input_sum if i < 0.0])
    output = np.where(input_sum < 0.0, Alpha * input_sum, input_sum)
    return output, input_sum

def leakyrelu_back_propagation(derror_wrt_output, input_sum, Alpha=0.25):
    """
        ������
                ��������Ԫ�����ƫ����dE/dIn = dE/dOut * dOut/dIn
                ���У�dOut/dIn ���Ǽ�����ĵ��� tanh'(x) = 1- x^2
                      dE/dOut ������Ԫ�����ƫ��
        ���룺
                derror_wrt_output:��������Ԫ�����ƫ��:��ʽ��5.7��
                input_sum: �����Ȩ��
        ���أ�
                derror_wrt_dinputs:�����������ƫ��
        """
    derror_wrt_dinput_sum = np.array(derror_wrt_output, copy=True)
    # derror_wrt_dinput_sum = np.squeeze([i * Alpha for i in derror_wrt_dinput_sum if i < 0.0])
    a = np.where(input_sum < 0.0)
    for i in a:
        derror_wrt_dinput_sum[i] = Alpha * derror_wrt_dinput_sum[i]
    return derror_wrt_dinput_sum

def activiated(activation_choose, input):
    """�����򼤻��װһ��"""
    if activation_choose == "sigmoid":
        return sigmoid(input)
    elif activation_choose == "relu":
        return relu(input)
    elif activation_choose == "tanh":
        return tanh(input)
    elif activation_choose == "leakyrelu":
        return leakyrelu(input)
    return sigmoid(input)#�൱��Ĭ��

def activated_back_propagation(activation_choose, derror_wrt_output, input):
    """
     �ѷ��򼤻����װһ��
    """
    if activation_choose == "sigmoid":
        return sigmoid_back_propagation(derror_wrt_output,input)
    elif activation_choose == "relu":
        return relu_back_propagation(derror_wrt_output, input)
    elif activation_choose == "tanh":
        return tanh_back_propagation(derror_wrt_output, input)
    elif activation_choose == "leakyrelu":
        return leakyrelu_back_propagation(derror_wrt_output, input)
    return sigmoid_back_propagation(derror_wrt_output, input)
