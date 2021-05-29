import cmath, math
import numpy as np


def is_power_of2(N):
    return (N & (N - 1) == 0) and N != 0



def cround(x):
    decimal_places = 3
    rounding_factor = 10 ** decimal_places

    x_real = math.floor(x.real * rounding_factor)/rounding_factor if x.real > 0 else math.ceil(x.real * rounding_factor)/rounding_factor
    x_imag = math.floor(x.imag * rounding_factor)/rounding_factor if x.imag > 0 else math.ceil(x.imag * rounding_factor)/rounding_factor
    return x_real + x_imag * 1j



def csum(li):
    ans = 0
    for item in li:
        ans += item
    return ans



def w(n, k, N):
    return cmath.exp(-2 * cmath.pi * 1j * n * k / N)



def fft(signal):

    signal_copy = signal.copy()

    while not is_power_of2(len(signal_copy)):
        signal_copy.append(0)

    N = len(signal_copy)

    if N == 2:
        return [signal_copy[0] + signal_copy[1], signal_copy[0] - signal_copy[1]]
    if N == 1:
        return [signal_copy[0]]

    X = [None] * N

    for k in range(int(N / 4)):

        X_even = csum([signal_copy[2 * n] * w(n, k, N / 2) for n in range(int(N / 2))])
        X_odd1 = csum([signal_copy[4 * n + 1] * w(n, k, N / 4) for n in range(int(N / 4))])
        X_odd3 = csum([signal_copy[4 * n + 3] * w(n, k, N / 4) for n in range(int(N / 4))])

        X_even2 = csum([signal_copy[2 * n] * w(n, k + N / 4, N / 2) for n in range(int(N / 2))])

        w1k = w(1, k, N)
        w3k = w(1, 3 * k, N)

        X[k] = X_even + (w1k * X_odd1 + w3k * X_odd3)
        X[k + int(N / 4)] = X_even2 - 1j * (w1k * X_odd1 - w3k * X_odd3)
        X[k + int(N / 2)] = X_even  - (w1k * X_odd1 + w3k * X_odd3)
        X[k + int(3 * N / 4)] = X_even2 + 1j * (w1k * X_odd1 - w3k * X_odd3)

    return [cround(item) for item in X]



def ifft(signal):

    signal_copy = signal.copy()

    while not is_power_of2(len(signal_copy)):
        signal_copy.append(0)

    N = len(signal_copy)

    if N == 2:
        return [cround(item) for item in [(signal_copy[0] + signal_copy[1]) / 2, (signal_copy[0] - signal_copy[1]) / 2]]
    if N == 1:
        return [signal_copy[0]]

    x = [None] * N

    for n in range(int(N / 2)):

        x_even = csum([signal_copy[2 * k] * w(-n, k, N / 2) for k in range(int(N / 2))]) / N
        x_odd1 = csum([signal_copy[4 * k + 1] * w(-n, k, N / 4) for k in range(int(N / 4))]) / N
        x_odd3 = csum([signal_copy[4 * k + 3] * w(-n, k, N / 4) for k in range(int(N / 4))]) / N

        w1n = w(-n, 1, N)
        w3n = w(-3 * n, 1, N)

        x[n] = x_even + (w1n * x_odd1 + w3n * x_odd3)
        x[n + int(N / 2)] = x_even - (w1n * x_odd1 + w3n * x_odd3)

    return [cround(item) for item in x]



base1 = [0, 4]
base2 = [4, -4]

test1 = [0, 2, 4, 6]
test2 = [(12+0j), (-4+4j), (-4+0j), (-4-4j)]

test3 = [0, 1, 2, 3, 4, 5, 6, 7]
test4 = [(28+0j), (-4+9.656j), (-4+3.999j), (-3.999+1.656j), (-4+0j), (-3.999-1.656j), (-4-4j), (-3.999-9.656j)]

test5 = [12, 5, 23, 6, 8, 17, 8, 2]
test6 = [(81+0j), (-7.313-9.343j), (-11-14j), (15.313+20.656j), (21+0j), (15.313-20.656j), (-11+13.999j), (-7.313+9.343j)]

print(ifft(test6))