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
    if np.abs(x_p - x) < 1e-7 and np.abs(y_p - y) < 1e-7:
        return True
    else:
        return False


def momentum(f, df, init, gamma=0.9, mu=0.1, max_iters=10e3):
    prev_tetta = np.array(init)
    v = mu * df(*init)
    cur_tetta = init - v
    iters = 1
    while not criterion(prev_tetta, cur_tetta) and iters < max_iters:
        iters += 1
        v = gamma * v + mu * df(*cur_tetta)
        prev_tetta = np.copy(cur_tetta)
        cur_tetta = cur_tetta - v
        print(f(*cur_tetta))
    result = {'fun': f(*cur_tetta), 'tettas': cur_tetta, 'iters': iters}

    return result


x, y = 0.7, 1.4
lr = 0.01
max_iter = 10 ** 3
print(momentum(f, df, (x, y), 1 - lr, lr, max_iter))