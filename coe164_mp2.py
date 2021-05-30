import cmath, math
import numpy as np


def is_power_of2(N):
    """A function that checks if an input number N
    is a power of 2

    Args:
        N (int): The number that may or may not be
        2 raised to the power of something.

    Returns:
        bool: either True or False.
    """    #
    return (N & (N - 1) == 0) and N != 0



def cround(x):
    """A function to floor numbers. Works for complex numbers.

    Args:
        x (complex): The complex number to be floored.

    Returns:
        complex: The floored complex number.
    """    
    decimal_places = 3
    rounding_factor = 10 ** decimal_places

    x_real = math.floor(x.real * rounding_factor)/rounding_factor if x.real > 0 else math.ceil(x.real * rounding_factor)/rounding_factor
    x_imag = math.floor(x.imag * rounding_factor)/rounding_factor if x.imag > 0 else math.ceil(x.imag * rounding_factor)/rounding_factor
    return x_real + x_imag * 1j



def csum(li):
    """A function to sum up complex numbers.

    Args:
        li (list): The input list of complex numbers.

    Returns:
        complex: The sum of complex numbers.
    """    
    ans = 0
    for item in li:
        ans += item
    return ans



def w(n, k, N):
    """Function that computes for the twiddle factor
    used in FFT and IFFT.

    Args:
        n (int): Discrete index of time-domain signal
        k (int): Discrete index of freq-domain signal
        N (int): Length of signal

    Returns:
        complex: The twiddle factor
    """    
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

        X_even = csum([signal_copy[2 * n] * w(n, k, N / 2) for n in range(N // 2)])
        X_odd1 = csum([signal_copy[4 * n + 1] * w(n, k, N / 4) for n in range(N // 4)])
        X_odd3 = csum([signal_copy[4 * n + 3] * w(n, k, N / 4) for n in range(N // 4)])

        X_even2 = csum([signal_copy[2 * n] * w(n, k + N / 4, N / 2) for n in range(N // 2)])

        w1k = w(1, k, N)
        w3k = w(1, 3 * k, N)

        X[k] = X_even + (w1k * X_odd1 + w3k * X_odd3)
        X[k + int(N / 4)] = X_even2 - 1j * (w1k * X_odd1 - w3k * X_odd3)
        X[k + int(N / 2)] = X_even  - (w1k * X_odd1 + w3k * X_odd3)
        X[k + int(3 * N / 4)] = X_even2 + 1j * (w1k * X_odd1 - w3k * X_odd3)

    return [cround(item) for item in X]



def ifft(signal):
    """A split-radix implementation of the inverse
    fast fourier transform (IFFT). 

    Args:
        signal (list): The list describing the frequency
        components of the signal.

    Returns:
        list: The list describing the freq-domain signal
        in the time domain.
    """    

    signal_copy = signal.copy()

    # append 0s to the signal if it is not a power of 2
    while not is_power_of2(len(signal_copy)):
        signal_copy.append(0)

    N = len(signal_copy)

    # base cases
    if N == 2:
        return [cround(item) for item in [(signal_copy[0] + signal_copy[1]) / 2, (signal_copy[0] - signal_copy[1]) / 2]]
    if N == 1:
        return [signal_copy[0]]

    x = [None] * N

    for n in range(int(N / 2)):

        # split the signal computations into even and two odd subsequences
        x_even = csum([signal_copy[2 * k] * w(-n, k, N / 2) for k in range(N // 2)]) / N
        x_odd1 = csum([signal_copy[4 * k + 1] * w(-n, k, N / 4) for k in range(N // 4)]) / N
        x_odd3 = csum([signal_copy[4 * k + 3] * w(-n, k, N / 4) for k in range(N // 4)]) / N

        # compute for the twiddle factors used in combining the signals
        w1n = w(-n, 1, N)
        w3n = w(-3 * n, 1, N)

        # merge the separated signals to compute for two time-domain answers
        x[n] = x_even + (w1n * x_odd1 + w3n * x_odd3)
        x[n + int(N / 2)] = x_even - (w1n * x_odd1 + w3n * x_odd3)

    # before returning the IFFT, round down each answer
    return [cround(item) for item in x]



base1 = [0, 4]
base2 = [4, -4]

test1 = [0, 2, 4, 6]
test2 = [(12+0j), (-4+4j), (-4+0j), (-4-4j)]

test3 = [0, 1, 2, 3, 4, 5, 6, 7]
test4 = [(28+0j), (-4+9.656j), (-4+3.999j), (-3.999+1.656j), (-4+0j), (-3.999-1.656j), (-4-4j), (-3.999-9.656j)]

test5 = [12, 5, 23, 6, 8, 17, 8, 2]
test6 = [(81+0j), (-7.313-9.343j), (-11-14j), (15.313+20.656j), (21+0j), (15.313-20.656j), (-11+13.999j), (-7.313+9.343j)]

test7 = [12, 35, 2, 35, 22, 16, 12, 74, 27, 34, 56, 12, 8, 12, 45, 7]
test8 = [(409+0j), (-83.554-3.449j), (62.033-5.949j), (-42.96+23.026j), (-46+31j), (42.66-128.009j), (-44.033-3.949j), (23.856-98.485j), (-41+0j), (23.856+98.485j), (-44.033+3.949j), (42.66+128.009j), (-46-31j), (-42.96-23.026j), (62.033+5.949j), (-83.554+3.449j)]

print(fft(test6))