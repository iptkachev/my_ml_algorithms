import numpy as np


def f(x, y):
    return (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2


def df(x, y):
    df_dx = lambda x, y: 2 * (y - 1) * (1.5 - x + x * y) + 2 * (y ** 2 - 1) * (2.25 - x + x * y ** 2) + \
                         2 * (y ** 3 - 1) * (2.625 - x + x * y ** 3)
    df_dy = lambda x, y: 2 * x * (1.5 - x + x * y) + 2 * 2 * x * y * (2.25 - x + x * y ** 2) + \
                         2 * 3 * x * y ** 2 * (2.625 - x + x * y ** 3)

    return np.array([df_dx(x, y), df_dy(x, y)])


def criterion(previous, current):
    x_p, y_p = previous
    x, y = current
    if np.abs(x_p - x) < 1e-8 and np.abs(y_p - y) < 1e-8:
        return True
    else:
        return False


def momentum(f, df, init: tuple, gamma=0.9, mu=0.1, max_iters=10e3):
    prev_tetta = np.array(init)
    v = mu * df(*prev_tetta)
    cur_tetta = prev_tetta - v
    iters = 1
    while not criterion(prev_tetta, cur_tetta) and iters < max_iters:
        iters += 1
        prev_tetta = np.copy(cur_tetta)
        v = gamma * v + mu * df(*prev_tetta)
        cur_tetta = cur_tetta - v
    result = {'fun': f(*cur_tetta), 'tettas': cur_tetta, 'iters': iters}

    return result


def nesterov_momentum(f, df, init: tuple, gamma=0.6, mu=0.03, max_iters=10e4):
    prev_tetta = np.array(init)
    v = mu * df(*prev_tetta)
    cur_tetta = prev_tetta - v
    iters = 1
    while not criterion(prev_tetta, cur_tetta) and iters < max_iters:
        iters += 1
        prev_tetta = np.copy(cur_tetta)
        v = gamma * v + mu * df(*(cur_tetta - gamma * v))
        cur_tetta = cur_tetta - v
    result = {'fun': f(*cur_tetta), 'tettas': cur_tetta, 'iters': iters}

    return result


def sgd(f, df, init: tuple, mu=0.03, max_iters=10e4):
    prev_tetta = np.array(init)
    v = mu * df(*prev_tetta)
    cur_tetta = prev_tetta - v
    iters = 1
    while not criterion(prev_tetta, cur_tetta) and iters < max_iters:
        iters += 1
        prev_tetta = np.copy(cur_tetta)
        v = mu * df(*prev_tetta)
        cur_tetta = cur_tetta - v
    result = {'fun': f(*cur_tetta), 'tettas': cur_tetta, 'iters': iters}

    return result


x, y = 0.7, 1.4
max_iter = 10 ** 4
print(momentum(f, df, (x, y), 0.85, 0.02, max_iter))
print(nesterov_momentum(f, df, (x, y), 0.85, 0.02, max_iter))
print(sgd(f, df, (x, y), 0.03, max_iter))

