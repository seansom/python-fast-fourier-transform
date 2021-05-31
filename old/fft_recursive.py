import cmath, math
import numpy as np


def npfft(li):
    return (np.fft.fft(li)).tolist()

def npifft(li):
    return (np.fft.ifft(li)).tolist()


def flatten_list(li):
    if isinstance(li[0], list):
        return [item for subli in li for item in subli]
    else:
        return li



def is_power_of2(N):
    return (N & (N - 1) == 0) and N != 0



def cround(x, return_real_only= False):
    decimal_places = 3
    rounding_factor = 10 ** decimal_places

    x_real = math.floor(x.real * rounding_factor)/rounding_factor if x.real > 0 else math.ceil(x.real * rounding_factor)/rounding_factor

    if return_real_only == True:
        return x_real

    x_imag = math.floor(x.imag * rounding_factor)/rounding_factor if x.imag > 0 else math.ceil(x.imag * rounding_factor)/rounding_factor
    return x_real + x_imag * 1j



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

    X_even = fft([signal[i] for i in range(N) if i % 2 == 0])
    X_odd1 = fft([signal[i] for i in range(N) if i != 0 and (i - 1) % 4 == 0])
    X_odd3 = fft([signal[i] for i in range(N) if i != 0 and (i - 3) % 4 == 0])

    w1k = [w(1, k, N) for k in range(N)]
    w3k = [w(1, 3 * k, N) for k in range(N)]

    sum_odd = [X_odd1[i] * w1k[i] + X_odd3[i] * w3k[i] for i in range(len(X_odd1))]
    diff_odd = [X_odd1[i] * w1k[i] - X_odd3[i] * w3k[i] for i in range(len(X_odd1))]

    E = len(X_even) // 2

    X0 = [X_even[i] + sum_odd[i] for i in range(E)]
    X1 = [X_even[i + E] - 1j * diff_odd[i] for i in range(E)]
    X2 = [X_even[i] - sum_odd[i] for i in range(E)]
    X3 = [X_even[i + E] + 1j * diff_odd[i] for i in range(E)]

    X = flatten_list([X0, X1, X2, X3])

    return [cround(item) for item in X]



def ifft(signal):

    signal_copy = signal.copy()

    while not is_power_of2(len(signal_copy)):
        signal_copy.append(0)

    N = len(signal_copy)

    # base cases
    if N == 2:
        return [cround(item) for item in [(signal_copy[0] + signal_copy[1]) / 2, (signal_copy[0] - signal_copy[1]) / 2]]
    if N == 1:
        return [signal_copy[0]]

    # split X into quadrants
    X0 = signal_copy[:N // 4]
    X1 = signal_copy[N // 4:N // 2]
    X2 = signal_copy[N // 2:N * 3 // 4]
    X3 = signal_copy[N * 3 // 4:N]

    # compute for sum_odd and diff_odd used
    sum_odd = [(X0[index] - X2[index]) / 2 for index in range(N // 4)]
    diff_odd = [(X3[index] - X1[index]) / 2j for index in range(N // 4)]

    # compute for the fourier-transformed even elements of x
    x_even = []
    x_even.append([X0[index] - sum_odd[index] for index in range(N // 4)]) 
    x_even.append([X1[index] + 1j * diff_odd[index] for index in range(N // 4)])
    x_even = flatten_list(x_even)

    # compute for the twiddle factors used to compute for sum and diff odd
    w1k = [w(1, k, N) for k in range(N)]
    w3k = [w(1, 3 * k, N) for k in range(N)]

    # compute for the fourier-transformed odd1 and odd3 elements of x
    x_odd1 = [(sum_odd[index] + diff_odd[index]) / (2 *  w1k[index]) for index in range(N // 4)]
    x_odd3 = [(sum_odd[index] - diff_odd[index]) / (2 *  w3k[index]) for index in range(N // 4)]

    # compute for the time-domain elements of x by recursively calling ifft()
    x_even = ifft(x_even)
    x_odd1 = ifft(x_odd1)
    x_odd3 = ifft(x_odd3)

    # reorder the subsignals into a single list of time-domain elements of x 
    x = [None] * N

    for index in range(N // 2):
        x[index * 2] = x_even[index]

    for index in range(N // 4):
        x[4 * index + 1] = x_odd1[index]
        x[4 * index + 3] = x_odd3[index]

    # round down each element of x before returning
    return [cround(item, return_real_only= True) for item in x]



base1 = [0, 4]
base2 = [4, -4]

test1 = [0, 2, 4, 6]
test2 = [(12+0j), (-4+4j), (-4+0j), (-4-4j)]

test3 = [0, 1, 2, 3, 4, 5, 6, 7]
test4 = [(28+0j), (-4+9.656854j), (-4+4j), (-4+1.656854j), (-4+0j), (-4-1.656854j), (-4-4j), (-4-9.656854j)]

test5 = [12, 5, 23, 6, 8, 17, 8, 2]
test6 = [(81+0j), (-7.313708-9.343145j), (-11-14j), (15.313708+20.656854j), (21+0j), (15.313708-20.656854j), (-11+14j), (-7.313708+9.343145j)]

test7 = [12, 35, 2, 35, 22, 16, 12, 74, 27, 34, 56, 12, 8, 12, 45, 7]
test8 = [(409+0j), (-83.554-3.449j), (62.033-5.949j), (-42.96+23.026j), (-46+31j), (42.66-128.009j), (-44.033-3.949j), (23.856-98.485j), (-41+0j), (23.856+98.485j), (-44.033+3.949j), (42.66+128.009j), (-46-31j), (-42.96-23.026j), (62.033+5.949j), (-83.554+3.449j)]

print(ifft(test4))