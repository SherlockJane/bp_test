# 2020/12/1 create by QAQ
# write a perceptron
# write a perceptron for 'and'
# realize that [1,1]->1 [1,0]->0 [0,1]->0 [0,0]->0
# w=w+delta(w) b=b+delta(b)
import functools as func

class Perceptron(object):
    def __init__(self, input_num, activator):
        self.activator = activator  # 激活函数
        self.weights = [0.0 for _ in range(input_num)]  # 权重向量
        self.bias = 0.0  # 偏置

    def __str__(self):
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    def prediction(self, input_vec):
        return self.activator(
            func.reduce(lambda a, b: a + b,
                        list(map(lambda w_x: w_x[0] * w_x[1], zip(input_vec, self.weights)))
                        , 0.0) + self.bias)  # 感知机输入的计算 w*x+b

    def train(self, input_vecs, labels, iteration, rate):
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)


    def _one_iteration(self, input_vecs, labels, rate):
        samples = zip(input_vecs, labels)
        for (input_vec, label) in samples:
            output = self.prediction(input_vec)
            self._update_weight(input_vec, output, label, rate)                # 更新权重


    def _update_weight(self, input_vec, output, label, rate):
        delta = label - output
        self.weights =  list(map(lambda w_x : w_x[0] + rate * delta * w_x[1],
                            zip(self.weights, input_vec)))
        # print(self.weights)
        self.bias = self.bias + rate * delta

def f(x):
    return 1 if x > 0 else 0

def get_training_dataset():
    input_vecs = [[1, 1], [0, 0], [1, 0], [0, 1]]
    labels = [1, 0, 0, 0]
    return input_vecs, labels

def train_and_perceptron():
    p = Perceptron(2, f)
    input_vecs, labels = get_training_dataset()
    p.train(input_vecs, labels, 10, 0.1)
    return p

if __name__ == '__main__':
    and_perceptron = train_and_perceptron()
    print(and_perceptron)
    print('1 and 1 = %d' % and_perceptron.prediction([1, 1]))
    print('1 and 0 = %d' % and_perceptron.prediction([1, 0]))
    print('0 and 1 = %d' % and_perceptron.prediction([0, 1]))
    print('0 and 0 = %d' % and_perceptron.prediction([0, 0]))

