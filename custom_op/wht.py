from math import ceil, log2
import torch as th


def walsh_func(n):
    if n == 0:
        return th.zeros((1, 1), dtype=th.int8)
    elif n == 2:
        return th.tensor([[1, 1], [1, -1]], dtype=th.int8)
    assert n % 2 == 0
    funcs_prev = walsh_func(n // 2)
    funcs = th.zeros((n, n), dtype=th.int8)
    funcs[: n // 2, : n // 2] = funcs_prev
    funcs[: n // 2, n // 2 :] = funcs_prev
    funcs[n // 2 :, : n // 2] = funcs_prev
    funcs[n // 2 :, n // 2 :] = -funcs_prev
    return funcs


def generate_gray(n):
    # Base case
    if n <= 0:
        return ["0"]
    if n == 1:
        return ["0", "1"]

    # Recursive case
    recAns = generate_gray(n - 1)
    mainAns = []
    # Append 0 to the first half
    for i in range(len(recAns)):
        s = recAns[i]
        mainAns.append("0" + s)
    # Append 1 to the second half
    for i in range(len(recAns) - 1, -1, -1):
        s = recAns[i]
        mainAns.append("1" + s)
    return mainAns


def generate_order(n):
    gray = generate_gray(int(log2(n)))
    order = [int(g[::-1], 2) for g in gray]
    return order


def hadamard_func(n):
    lgn = log2(n)
    m = 2 ** ceil(lgn)
    walsh = walsh_func(m)
    order = generate_order(m)
    hadamard = walsh[:, order]
    if m != n:
        hadamard = hadamard[:n, :n]
    return hadamard
