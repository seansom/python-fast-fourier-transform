import sys, cmath, math, re
import mpmath as mp
# import numpy as np



# def npfft(li):
#     return (np.fft.fft(li)).tolist()



# def npifft(li):
#     return (np.fft.ifft(li)).tolist()


def memoize(func):
    memo = {}

    def wrapper(*args):

        if args not in memo:
            memo[args] = func(*args)

        return memo[args]

    return wrapper



def flatten_list(li):
    if isinstance(li[0], list):
        return [item for subli in li for item in subli]
    else:
        return li



@memoize
def is_power_of2(N):
    return (N & (N - 1) == 0) and N != 0



def cround(x, return_real_only= False):

    decimal_places = 6
    rounding_factor = 10 ** decimal_places

    x_real = float(mp.fdiv(mp.nint((mp.fmul(mp.re(x), rounding_factor))), rounding_factor))
    x_imag = float(mp.fdiv(mp.nint((mp.fmul(mp.im(x), rounding_factor))), rounding_factor))

    if return_real_only:
        return x_real

    else:
        return complex(x_real, x_imag)


@memoize
def w(n, k, N):
    return mp.exp(mp.mpc(-2 * mp.pi * (1j) * n * k) / N)



def ifft_helper(signal):

    signal_copy = signal.copy()

    while not is_power_of2(len(signal_copy)):
        signal_copy.append(0)

    N = len(signal_copy)

    if not isinstance(signal_copy[0], mp.mpc):
        signal_copy = [mp.mpc(item) for item in signal_copy]

    # base cases
    if N == 2:
        return [(signal_copy[0] + signal_copy[1]) / 2, (signal_copy[0] - signal_copy[1]) / 2]
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
    x_even = ifft_helper(x_even)
    x_odd1 = ifft_helper(x_odd1)
    x_odd3 = ifft_helper(x_odd3)

    # reorder the subsignals into a single list of time-domain elements of x 
    x = [None] * N

    for index in range(N // 2):
        x[index * 2] = x_even[index]

    for index in range(N // 4):
        x[4 * index + 1] = x_odd1[index]
        x[4 * index + 3] = x_odd3[index]

    return x



def ifft(signal):
    # ifft() only rounds the final answer, the transformation itself is handled by ifft_helper()
    return [math.floor(cround(item, return_real_only= True)) for item in ifft_helper(signal)]



def main():
    lines = []

    line = sys.stdin.readline()
    while line:
        lines.append(line)
        line = sys.stdin.readline()

    signals_num = int(lines.pop(0)[0])

    answers = []

    for index in range(signals_num):

        signal = lines[index]

        # pop out the time signal length
        time_signal_length = re.search(r'\d+', signal).group()
        signal = signal.replace(time_signal_length, '', 1)

        # pop out the freq signal length
        freq_signal_length = re.search(r'\d+', signal).group()
        signal = signal.replace(freq_signal_length, '', 1)
        
        # convert freq elements string into a complex list
        signal = signal.replace(' ', '')
        signal_regex = re.compile(r'[\+-][\d]+.[\d]+[\+-][\d]+.[\d]+j')
        signal = [complex(item) for item in signal_regex.findall(signal)]

        time_signal = ifft(signal)
        ans = [time_signal_length]

        for i in range(int(time_signal_length)):
            ans.append(str(time_signal[i]))

        answers.append(' '.join(ans))

    print(signals_num)
    for ans in answers:
        print(ans)



# base1 = [0, 4]
# base2 = [4, -4]

# test1 = [0, 2, 4, 6]
# test2 = [(12+0j), (-4+4j), (-4+0j), (-4-4j)]

# test3 = [0, 1, 2, 3, 4, 5, 6, 7]
# test4 = [(28+0j), (-4+9.656854j), (-4+4j), (-4+1.656854j), (-4+0j), (-4-1.656854j), (-4-4j), (-4-9.656854j)]

# test5 = [12, 5, 23, 6, 8, 17, 8, 2]
# test6 = [(81+0j), (-7.313708-9.343145j), (-11-14j), (15.313708+20.656854j), (21+0j), (15.313708-20.656854j), (-11+14j), (-7.313708+9.343145j)]

# test7 = [12, 35, 2, 35, 22, 16, 12, 74, 27, 34, 56, 12, 8, 12, 45, 7]
# test8 = [(409+0j), (-83.554-3.449j), (62.033-5.949j), (-42.96+23.026j), (-46+31j), (42.66-128.009j), (-44.033-3.949j), (23.856-98.485j), (-41+0j), (23.856+98.485j), (-44.033+3.949j), (42.66+128.009j), (-46-31j), (-42.96-23.026j), (62.033+5.949j), (-83.554+3.449j)]

# test9 = [1.234567, 7.654321, 1.001000, 0]
# test10 = [(9.889888+0j), (0.23356700000000008-7.654321j), (-5.418754000000001+0j), (0.23356700000000008+7.654321j)]

# test11 = [1, 2, 3, 4, 5, 0, 0, 0]
# test12 = [(15+0j), (-5.414213562373095-7.242640687119286j), (3+2j), (-2.585786437626905-1.2426406871192857j), (3+0j), (-2.585786437626905+1.2426406871192857j), (3-2j), (-5.414213562373095+7.242640687119286j)]

# print(ifft(npfft(test1)))
# print(ifft(npfft(test3)))
# print(ifft(npfft(test5)))
# print(ifft(npfft(test7)))
# print(ifft(npfft(test9)))


if __name__ == '__main__':
    main()