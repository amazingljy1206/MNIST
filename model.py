import numpy as np

class FCN(object):
    def __init__(self, num_input, num_output):  # 全连接层初始化
        self.num_input = num_input
        self.num_output = num_output
    def init_param(self, std = 0.01):  # 参数初始化
        self.weight = np.random.normal(loc=0.0, scale = std, size = (self.num_input, self.num_output))
        self.bias = np.zeros([1, self.num_output])
    def forward(self, input):  # 前向传播计算
        self.input = input
        self.output = np.matmul(input, self.weight) + self.bias
        return self.output
    def backward(self, top_diff):  # 反向传播的计算
        self.d_weight = np.dot(self.input.T, top_diff)
        self.d_bias = np.sum(top_diff, axis=0)
        bottom_diff = np.dot(top_diff, self.weight.T)
        return bottom_diff
    def update_param(self, lr):  # SGD优化
        self.weight = self.weight - lr*self.d_weight
        self.bias = self.bias - lr*self.d_bias
    def load_param(self, weight, bias):  # 参数加载
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias
    def save_param(self):  # 参数保存
        return self.weight, self.bias

class ReLU(object):
    def __init__(self):
        pass
    def forward(self, input):  # 前向传播的计算
        self.input = input
        output = np.maximum(0, input)
        return output
    def backward(self, top_diff):  # 反向传播的计算
        bottom_diff = top_diff
        bottom_diff[self.input<0] = 0
        return bottom_diff

class SoftmaxCrossEntropy(object):
    def __init__(self):
        pass
    def forward(self, input):  # 前向传播的计算
        input_max = np.max(input, axis=1, keepdims=True)
        input_exp = np.exp(input - input_max)
        self.prob = input_exp / np.sum(input_exp, axis=1, keepdims=True)
        return self.prob
    def get_loss(self, label):   # 计算损失
        self.batch_size = self.prob.shape[0]
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), label] = 1.0
        loss = -np.sum(np.log(self.prob) * self.label_onehot) / self.batch_size
        return loss
    def backward(self):  # 反向传播的计算
        bottom_diff = (self.prob - self.label_onehot) / self.batch_size
        return bottom_diff