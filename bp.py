# 2020/12/2 create by QAQ
# test for Back Propagation 单层
import random
import functools as func
import numpy as np


def sigmoid(x):
    # TODO: Implement sigmoid function
    return 1/(1 + np.exp(-x))


class Node(object):
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.upstream = []
        self.downstream = []
        self.output = 0
        self.delta = 0

    def set_output(self, output):
        self.output = output

    def append_downstream_connection(self, conn):
        self.downstream.append(conn)

    def append_upstream_connection(self, conn):
        self.upstream.append(conn)

    def calc_output(self):
        output = func.reduce(lambda ret_conn: ret_conn[0] + ret_conn[1].upstream_node.output * ret_conn[1].weight, self.upstream, 0)
        self.output = sigmoid(output)   # conn为connection

    def calc_hidden_layer_delta(self):
        down_stream_delta = func.reduce(lambda ret_conn: ret_conn[0] + ret_conn[1].downstream_node.delta * ret_conn[1].weight, self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * down_stream_delta

    def calc_output_layer_delta(self, label):
        self.delta = self.output * (1 - self.output) * (label - self.output)

    def __str__(self):
        node_str = '%u-%u: output :%f delta %f' % (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = func.reduce(lambda ret_conn: ret_conn[0] + '\n\t' + str(ret_conn[1]), self.downstream, '')
        upstream_str = func.reduce(lambda  ret_conn: ret_conn[0] + '\n\t' + str(ret_conn[1]), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str


class ConstNode(object):
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.output = 1
        self.delta = 0

    def append_downstream_connection(self, conn):
        self.downstream.append(conn)

    def calc_hidden_layer_delta(self):
        downstream_delta = func.reduce(lambda ret_conn: ret_conn[0] + ret_conn[1].downstream_node.delta * ret_conn[1].weight, self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def __str__(self):
        node_str = '%u-%u: output :1 ' % (self.layer_index, self.node_index)
        downstream_str = func.reduce(lambda ret_conn: ret_conn[0] + '\n\t' + str(ret_conn[1]), self.downstream, '')
        return node_str + '\n\tdownstream:' + downstream_str


class Layer(object):
    def __init__(self, layer_index, node_count):
        self.layer_index = layer_index
        self.nodes = []
        for i in range(node_count):
            self.nodes.append(Node(layer_index, i))
        self.nodes.append(ConstNode(layer_index, node_count))

    def set_output(self, data):
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])

    def calc_output(self):
        for node in self.nodes[:-1]:
            node.calc_output()

    def dump(self):
        for node in self.nodes:
            print(node)


class Connection(object):
    def __init__(self, upstream_node, downstream_node):
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.weight = random.uniform(-0.1, 0.1)
        self.gradient = 0.0  # 记录w的增量

    def calc_gradient(self):
        self.gradient = self.downstream_node.delta * self.downstream_node.output

    def get_gradient(self):
        return self.gradient

    def update_weight(self, rate):
        self.calc_gradient()
        self.weight += rate * self.gradient

    def __str__(self):
        return '%u-%u -> %u-%u =%f' % (self.upstream_node.layer_index, self.upstream_node.node_index, self.downstream_node.layer_index, self.downstream_node.node_index, self.weight)

class Connections(object):
    def __init__(self):
        self.connections = []

    def add_connection(self, connection):
        self.connections.append(connection)

    def dump(self):
        for conn in self.connections:
            print(conn)


class Network(object):
    def __init__(self, layers): # layers是一个表示每层点数的数组
        self.connections = Connections()
        self.layers = []
        layer_count = len(layers)
        node_count = 0
        for i in range(layer_count):
            self.layers.append(Layer(i, layers[i]))
        for layer in range(layer_count - 1):
            connections = [Connection(upstream_node, downstream_node)
                           for upstream_node in self.layers[layer].nodes
                           for downstream_node in self.layers[layer + 1].nodes[:-1]]
            for conn in connections:
                self.connections.add_connection(conn)
                conn.upstream_node.append_downstream_connection(conn)
                conn.downstream_node.append_ipstream_connection(conn)

    def train(self, labels, dataset, rate, iteration):
        for i in range(iteration):
            for d in range(len(dataset)):
                self.train_one_sample(labels[d], dataset[d], rate)

    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.cal_delta(label)
        self.update_weight(rate)

    def predict(self, sample):
        self.layers[0].set_output(sample)
        for i in range(1, len(self.layers)):
            self.layers[i].calc_output()
        return map(lambda node: node.output, self.layers[-1].nodes[:-1])

    def cal_delta(self, label):
        output_nodes = self.layers[-1].nodes
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta(label[i])
        for layer in self.layers[-2::-1]: # 起始位置、结束位置、步长，即倒数第二个元素（含）取到第一个元素（不含）
            for node in layer.nodes:
                node.calc_hidden_layer_delta()

    def update_weight(self, rate):
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.update_weight(rate)

    def calc_gradient(self):
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()

    def dump(self):
        for layer in self.layers:
            layer.dump()


# test for weight
def gradient_check(network, sample_feature, sample_label):
    '''
    梯度检查
    network: 神经网络对象
    sample_feature: 样本的特征
    sample_label: 样本的标签
    '''
    # 计算网络误差
    network_error = lambda vec1, vec2: \
            0.5 * func.reduce(lambda a, b: a + b,
                      map(lambda v: (v[0] - v[1]) * (v[0] - v[1]),
                          zip(vec1, vec2)))
    # 获取网络在当前样本下每个连接的梯度
    network.get_gradient(sample_feature, sample_label)
    # 对每个权重做梯度检查
    for conn in network.connections.connections:
        # 获取指定连接的梯度
        actual_gradient = conn.get_gradient()
        # 增加一个很小的值，计算网络的误差
        epsilon = 0.0001
        conn.weight += epsilon
        error1 = network_error(network.predict(sample_feature), sample_label)
        # 减去一个很小的值，计算网络的误差
        conn.weight -= 2 * epsilon # 刚才加过了一次，因此这里需要减去2倍
        error2 = network_error(network.predict(sample_feature), sample_label)
        # 根据式6计算期望的梯度值
        expected_gradient = (error2 - error1) / (2 * epsilon)
        # 打印
        print('expected gradient: \t%f\nactual gradient: \t%f' % (
            expected_gradient, actual_gradient))

if __name__ == '__main__':
    net = Network([2, 4, 5])
     
