# 2020/12/2 create by QAQ
# test for BGD, SGD and MBGD

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def quadratic(x, y):
    return 0.5 * x ** 2 + 0.1 * y ** 2


def quadratic_grad(x, y):
    return np.array([x, 0.2 * y])


max_eta = 1.66
min_eta = 0.001


def strategy1(grad, last_grad, iters):
    """
    固定步长 min_eta 。
    """
    global max_eta, min_eta
    eta = min_eta
    return -eta * grad


def strategy2(grad, last_grad, iters):
    """
    固定步长 max_eta 。
    """
    global max_eta, min_eta
    return -max_eta * grad


def strategy3(grad, last_grad, iters):
    """
    固定步长 max_eta ，以 0.1 为系数加入冲量。
    """
    global max_eta, min_eta
    return -max_eta * ((1 - 0.1) * grad + 0.1 * last_grad)


def strategy4(grad, last_grad, iters):
    """
    初始步长为 max_eta ，以 0.9 为衰减因子衰减步长。保障步长大于等于 min_eta 。
    """
    global max_eta, min_eta

    if not iters:
        eta = max_eta
    else:
        eta = min_eta if max_eta * 0.9 ** iters <= min_eta else max_eta * 0.9 ** iters

    return -eta * grad


def strategy5(grad, last_grad, iters):
    """
    根据梯度的大小。在梯度大的地方减小步长，在梯度小的地方增加步长。保障步长大于等于 min_eta 。
    具体计算方法是：步长＝max_eta / e^(magnitude/10.0) 。
    e 是自然对数的底，magnitude 是当前梯度的长度。
    """
    global max_eta, min_eta

    magnitude = np.sqrt(np.dot(grad, grad))
    coe = np.power(np.e, magnitude / 10.0)
    if coe > 0:
        eta = max_eta / coe
    else:
        eta = min_eta

    eta = min_eta if eta <= min_eta else eta
    return -eta * grad


def strategy6(grad, last_grad, iters):
    """
    根据当前梯度与上一位置的梯度交角调整步长。
    交角大，认为地形崎岖变化大，缩小步长。若交角小，认为地形变化不大，增大步长。
    具体步长公式为：步长 = max_eta * (cos_theta ＋ 1.0001) / 2.0001 。
    cos_theta 是本次梯度与上一次梯度的交角余弦值。加上 1.0001 是将系数调整为正数。
    """
    global max_eta, min_eta

    l_grad = np.sqrt(np.dot(grad, grad))
    l_last_grad = np.sqrt(np.dot(last_grad, last_grad))

    if l_grad > 0 and l_last_grad > 0:
        cos_theta = np.dot(grad, last_grad) / (l_grad * l_last_grad)
        eta = max_eta * (cos_theta + 1.001) / 2.0001
    else:
        eta = max_eta

    eta = min_eta if eta <= min_eta else eta
    return -eta * grad


def draw_chart(fun, path, ax):
    x, y = np.meshgrid(np.arange(-8.0, 8.0, 0.1), np.arange(-8.0, 8.0, 0.1))
    z = fun(x, y)

    ax.plot_surface(x, y, z, rstride=1, cstride=1, alpha=0.5, cmap=cm.coolwarm)
    ax.contourf(x, y, z, zdir='z', offset=-10.0, cmap=cm.coolwarm)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_zlim([-10.0, 40.0])

    if path is not None:
        ax.plot(path[0], path[1], [-10.0] * len(path[0]), c="#b22222", linewidth=0.8)
        ax.scatter(path[0][-1], path[1][-1], [-10.0], c="#b22222")


def gradient_descent(init_position, gradient_fun, step_fun, tolerance=1e-4, max_iters=None):
    x = [init_position[0]]
    y = [init_position[1]]
    iters = 0
    last_grad = np.array([0.0, 0.0])

    while True:
        cx = x[-1]
        cy = y[-1]

        grad = gradient_fun(cx, cy)
        step = step_fun(grad, last_grad, iters)

        x.append(cx + step[0])
        y.append(cy + step[1])

        last_grad = grad
        iters += 1
        magnitude = np.sqrt(np.dot(grad, grad))

        if magnitude < tolerance or (max_iters is not None and iters >= max_iters):
            break

    return {"final_pos": [x[-1], y[-1]], "iters": iters, "final_grad": grad, "path": [x, y]}


fig = plt.figure(figsize=(20, 30))

strategies = [strategy1, strategy2, strategy3, strategy4, strategy5, strategy6]

for step_func, index in zip(strategies, np.arange(1, len(strategies) + 1)):
    ax = fig.add_subplot(3, 2, index, projection="3d")
    result = gradient_descent([-6.0, -6.0], quadratic_grad, step_func)
    draw_chart(quadratic, result["path"], ax)
    ax.set_title("strategy: {:d} loops: {:,}".format(index, result["iters"]))
    print("running strategy {:d}".format(index))

plt.savefig("strategies.png")
plt.cla()
plt.clf()
plt.close()
