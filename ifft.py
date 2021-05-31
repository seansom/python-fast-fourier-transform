import sys, cmath, math, re, decimal
import mpmath as mp

# import numpy as np

# def npfft(li):
#     return (np.fft.fft(li)).tolist()

# def npifft(li):
#     return (np.fft.ifft(li)).tolist()


class hpc:
    """A high precision complex class utillizing the
    standard library Decimal class.
    """    

    def __init__(self, real, imag= 0):

        if isinstance(real, hpc):
            self.real = real.real
            self.imag = real.imag

        # i.e. hpc(1 + 2j)
        elif isinstance(real, complex):
            self.real = decimal.Decimal(real.real)
            self.imag = decimal.Decimal(real.imag)

        # i.e. hpc('1 + 2j')
        elif isinstance(real, str):
            real = real.replace(' ', '')
            hpc_regex = re.compile(r'([\+-]?[\d]+.?[\d]*)([\+-][\d]+.?[\d]*)j')
            real, imag = hpc_regex.findall(real)[0]

            self.real = decimal.Decimal(real)
            self.imag = decimal.Decimal(imag)

        # i.e. hpc(1, 2) or hpc('1', '2')
        else:
            self.real = decimal.Decimal(real)
            self.imag = decimal.Decimal(imag)



    def __repr__(self):
        """Dunder method that gives the string
        representation of the hpc.

        Returns:
            str: string representation of the hpc
        """        
        if self.imag == 0:
            return f'{self.real}'

        if self.real == 0:
            return f'{self.imag}j'

        if self.imag < 0:
            return f'{self.real} - {-self.imag}j'

        return f'{self.real} + {self.imag}j'



    def __add__(self, other):
        """Dunder method that overloads the `+` operator.

        Args:
            other (hpc, complex, string, or num class): 
            The number to be added to the hpc self

        Returns:
            hpc: The sum of self and other
        """        

        if not isinstance(other, hpc):
            other = hpc(other)

        ans_real = self.real + other.real
        ans_imag = self.imag + other.imag
        return hpc(ans_real, ans_imag)



    def __sub__(self, other):
        """Dunder method that overloads the `-` operator.

        Args:
            other (hpc, complex, string, or num class): 
            The number to be subracted to the hpc self

        Returns:
            hpc: The difference of self and other
        """

        if not isinstance(other, hpc):
            other = hpc(other)

        ans_real = self.real - other.real
        ans_imag = self.imag - other.imag
        return hpc(ans_real, ans_imag)



    def __mul__(self, other):
        """Dunder method that overloads the `*` operator.

        Args:
            other (hpc, complex, string, or num class): 
            The number to be multiplied to the hpc self

        Returns:
            hpc: The product of self and other
        """

        if not isinstance(other, hpc):
            other = hpc(other)

        ans_real = (self.real * other.real) - (self.imag * other.imag)
        ans_imag = (self.real * other.imag) + (self.imag * other.real)
        return hpc(ans_real, ans_imag)



    def __truediv__(self, other):
        """Dunder method that overloads the `/` operator.

        Args:
            other (hpc, complex, string, or num class): 
            The number to be divide the hpc self

        Returns:
            hpc: The quotient of self and other
        """

        if not isinstance(other, hpc):
            other = hpc(other)

        denom = other.real ** 2 + other.imag ** 2

        if not denom:
            return hpc(self.imag / other.imag)

        ans_real = ((self.real * other.real) + (self.imag * other.imag)) / denom
        ans_imag = ((self.imag * other.real) - (self.real * other.imag)) / denom

        return hpc(ans_real, ans_imag)



    def re(self):
        """Method that returns the real part of the hpc.

        Returns:
            Decimal: real part of the hpc
        """        
        return self.real
    


    def im(self):
        """Method that returns the imaginary part of the hpc.

        Returns:
            Decimal: imaginary part of the hpc
        """ 
        return self.imag
    


def memoize(func):
    """Function decorator to provide memoization. If func
    is called, wrapper is called instead.

    Args:
        func (function): The function to be memoized

    Returns:
        wrapper: The wrapper function
    """    
    memo = {}

    def wrapper(*args):

        if args not in memo:
            memo[args] = func(*args)

        return memo[args]

    return wrapper



def flatten_list(li):
    """Flattens a 2d list into a single list.

    Args:
        li (2d list): The 2d list to be flattened

    Returns:
        1d list: The flattened list
    """    
    if isinstance(li[0], list):
        return [item for subli in li for item in subli]
    else:
        return li



@memoize
def is_power_of2(N):
    """Checks if an input N is a power of 2.

    Args:
        N (int): The int in question

    Returns:
        bool: True or False
    """    
    return (N & (N - 1) == 0) and N != 0



def hpcround(x, decimal_places= 6, return_real_only= False):
    """Function that rounds off hpc objects and converts it into a 
    complex object.

    Args:
        x (hpc): The hpc to be rounded off
        decimal_places (int, optional): The number of decimal places 
        desired. Defaults to 6.
        return_real_only (bool, optional): Boolean option to return only 
        real part of the hpc. Defaults to False.

    Returns:
        complex: The rounded off hpc as a complex object
    """    

    rounding_factor = 10 ** decimal_places

    x_real = x.re() * rounding_factor
    real_delta = x_real - int(x_real)
    
    if real_delta >= 0.5 or -0.5 < real_delta <= 0:
        x_real = math.ceil(x_real)
    else:
        x_real = math.floor(x_real)

    x_real = x_real / rounding_factor

    if return_real_only:
        return x_real

    x_imag = x.im() * rounding_factor
    imag_delta = x_imag - int(x_imag)

    if imag_delta >= 0.5 or -0.5 < imag_delta <= 0:
        x_imag = math.ceil(x_imag)
    else:
        x_imag = math.floor(x_imag)

    x_imag = x_imag / rounding_factor

    return complex(x_real, x_imag)



@memoize
def w(n, k, N):
    """Function that computes for the twiddle factor
    used in fourier transformations.

    Args:
        n (int): index of time-domain element
        k (int): index of freq-domain element
        N (int): Number of total elements

    Returns:
        hpc: e ^ (-2πnk / N)
    """    
    return hpc(cmath.exp((-2 * cmath.pi * (1j) * n * k) / N))



def ifft_helper(signal):
    """Helper function to ifft() that solves for the inverse
    fast fourier transform of a freq-domain signal.

    Args:
        signal (list): The list (hpc, complex, or str) that
        represents the freq-domain elements of the signal to 
        be transformed.

    Returns:
        list: A list of hpc objects that represents the 
        time-domain elements of the transformed signal.
    """

    signal_copy = signal.copy()

    while not is_power_of2(len(signal_copy)):
        signal_copy.append(0)

    N = len(signal_copy)

    if not isinstance(signal_copy[0], hpc):
        signal_copy = [hpc(item) for item in signal_copy]

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
    x_even.append([X1[index] + diff_odd[index] * 1j for index in range(N // 4)])
    x_even = flatten_list(x_even)

    # compute for the twiddle factors used to compute for sum and diff odd
    w1k = [w(1, k, N) for k in range(N)]
    w3k = [w(1, 3 * k, N) for k in range(N)]

    # compute for the fourier-transformed odd1 and odd3 elements of x
    x_odd1 = [(sum_odd[index] + diff_odd[index]) / (w1k[index] *  2) for index in range(N // 4)]
    x_odd3 = [(sum_odd[index] - diff_odd[index]) / (w3k[index] *  2) for index in range(N // 4)]

    # compute for the time-domain elements of x by recursively calling ifft_helper()
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
    """The client function that returns the inverse
    fast fourier transform of a freq-domain signal. The 
    function only floors the elements of the transformed 
    signal, and the transformation itself is handled by 
    ifft_helper().

    Args:
        signal (list): The list (hpc, complex, or str) that
        represents the freq-domain elements of the signal to 
        be transformed.

    Returns:
        list: A list of integers that represents the 
        time-domain elements of the transformed signal.
    """

    return [int(hpcround(item, decimal_places= 1, return_real_only= True)) for item in ifft_helper(signal)]



def main():
    """The main function that handles the reading and writing
    to stdin and stdout.
    """    
    lines = []

    line = sys.stdin.readline()
    while line:
        lines.append(line)
        line = sys.stdin.readline()

    signals_num = int(lines.pop(0)[0])

    answers = [signals_num]

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

        signal = [hpc(item) for item in signal_regex.findall(signal)]

        time_signal = ifft(signal)
        ans = [time_signal_length]

        for i in range(int(time_signal_length)):
            ans.append(str(time_signal[i]))

        answers.append(' '.join(ans))

    for ans in answers:
        print(ans)



if __name__ == '__main__':
    main()



# test1 = [0, 2, 4, 6]
# test2 = [(12+0j), (-4+4j), (-4+0j), (-4-4j)]

# test3 = [0, 1, 2, 3, 4, 5, 6, 7]
# test4 = [(28+0j), (-4+9.656854j), (-4+4j), (-4+1.656854j), (-4+0j), (-4-1.656854j), (-4-4j), (-4-9.656854j)]

# test5 = [12, 5, 23, 6, 8, 17, 8, 2]
# test6 = [(81+0j), (-7.313708-9.343145j), (-11-14j), (15.313708+20.656854j), (21+0j), (15.313708-20.656854j), (-11+14j), (-7.313708+9.343145j)]

# test7 = [12, 35, 2, 35, 22, 16, 12, 74, 27, 34, 56, 12, 8, 12, 45, 7]
# test8 = [(409+0j), (-83.554-3.449j), (62.033-5.949j), (-42.96+23.026j), (-46+31j), (42.66-128.009j), (-44.033-3.949j), (23.856-98.485j), (-41+0j), (23.856+98.485j), (-44.033+3.949j), (42.66+128.009j), (-46-31j), (-42.96-23.026j), (62.033+5.949j), (-83.554+3.449j)]

# test9 = [1.234567, 7.654321, 1.001001, 0.000001]
# test10 = [(9.889888+0j), (0.23356700000000008-7.654321j), (-5.418754000000001+0j), (0.23356700000000008+7.654321j)]

# test11 = [1, 2, 3, 4, 5, 0, 0, 0]
# test12 = [(15+0j), (-5.414213562373095-7.242640687119286j), (3+2j), (-2.585786437626905-1.2426406871192857j), (3+0j), (-2.585786437626905+1.2426406871192857j), (3-2j), (-5.414213562373095+7.242640687119286j)]

# print(ifft(npfft(test1)))
# print(ifft(npfft(test3)))
# print(ifft(npfft(test5)))
# print(ifft(npfft(test7)))
# print(ifft(npfft(test9)))
# print(ifft(npfft(test11)))