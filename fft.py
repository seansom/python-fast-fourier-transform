import sys, math, re, decimal


class hpc:
    """A high precision complex class utilizing the
    standard library Decimal class.

    Args:
        x (complex, str, or number): An hpc may be
        initialized by a single complex or str object,
        or a single or pair of number objects.
        y (number, optional): The number representing
        the imaginary part of the complex number. 
        Defaults to None or 0 if x is of a number type.

    >>> hpc('1.2 - 2.2j')
    '1.2-2.2j'
    """

    def __init__(self, x, y= None):

        if y is None:

            if isinstance(x, hpc):
                self.real = x.real
                self.imag = x.imag

            # i.e. hpc(1 + 2j)
            elif isinstance(x, complex):
                self.real = decimal.Decimal(x.real)
                self.imag = decimal.Decimal(x.imag)

            # i.e. hpc('1 + 2j') or hpc('-2.0') or hpc('3.2j')
            elif isinstance(x, str):
                x = x.replace(' ', '')

                real_imag_regex = re.compile(r'^([\+-]?[\d]+\.?[\d]*)([\+-][\d]+\.?[\d]*)j$')
                real_only_regex = re.compile(r'^([\+-]?[\d]+\.?[\d]*)$')
                imag_only_regex = re.compile(r'^([\+-]?[\d]+\.?[\d]*)j$')

                if match := real_imag_regex.findall(x):
                    real, imag = match[0]
                    self.real = decimal.Decimal(real)
                    self.imag = decimal.Decimal(imag)

                elif match:= real_only_regex.findall(x):
                    self.real = decimal.Decimal(match[0])
                    self.imag = decimal.Decimal(0)

                elif match:= imag_only_regex.findall(x):
                    self.real = decimal.Decimal(0)
                    self.imag = decimal.Decimal(match[0])

                else:
                    raise ValueError('Malformed string in hpc constructor.')

            # i.e. hpc(1.23)
            else:
                try:
                    self.real = decimal.Decimal(x)
                    self.imag = decimal.Decimal(0)
                except:
                    raise ValueError('Invalid hpc constructor arguments.')


        # i.e. hpc(1, 2) or hpc('1', '2') or hpc('1.', -2.32)
        else:
            try:
                self.real = decimal.Decimal(x)
                self.imag = decimal.Decimal(y)
            except:
                raise ValueError('Invalid hpc constructor arguments.')


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

    
    def __complex__(self):
        """Overload of type conversion from hpc to complex class.
        Note: type conversion to complex reduces decimal places
        precision to 18.

        Returns:
            complex: The complex representation of the hpc.
        """        
        return complex(self.real, self.imag)


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
            return f'{self.real}-{-self.imag}j'

        return f'{self.real}+{self.imag}j'


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
    

    def  __pow__(self, other):
        """Dunder method that overloads the `**` operator.
        Currently does not ensure high precision.

        Args:
            other (hpc): The hpc representing the exponent

        Returns:
            hpc: self raised to the power of other
        """
        return hpc(complex(self.real, self.imag) ** complex(other.real, other.imag))


    # Since addition and multiplication are commutative,
    # set reverse operations to be the same
    __radd__ = __add__
    __rmul__ = __mul__


    def  __rsub__(self, other):
        # reverse subtraction
        other = hpc(other)
        return hpc.__sub__(other, self)

    
    def __rtruediv__(self, other):
        # reverse division
        other = hpc(other)
        return hpc.__truediv__(other, self)


    def __rpow__(self, other):
        # reverse power
        other = hpc(other)
        return hpc.__pow__(other, self)


def hpc_round(x, decimal_places=28):
    """Function that rounds off hpc objects to a specified number of 
    decimal places.

    Args:
        x (hpc): The hpc to be rounded off
        decimal_places (int, optional): The number of decimal places 
        desired. Defaults to 28.

    Returns:
        hpc: The rounded off hpc object.
    
    >>> hpc_round(hpc('1.234567-5.4321j'), decimal_places=3)
    '1.235-5.432j'
    """
    rounding_factor = 10 ** decimal_places

    x_real = x.re() * rounding_factor
    real_delta = x_real - int(x_real)
    x_imag = x.im() * rounding_factor
    imag_delta = x_imag - int(x_imag)
    
    if real_delta >= 0.5 or -0.5 < real_delta <= 0:
        x_real = decimal.Decimal(math.ceil(x_real))
    else:
        x_real = decimal.Decimal(math.floor(x_real))

    if imag_delta >= 0.5 or -0.5 < imag_delta <= 0:
        x_imag = decimal.Decimal(math.ceil(x_imag))
    else:
        x_imag = decimal.Decimal(math.floor(x_imag))

    x_real = x_real / rounding_factor
    x_imag = x_imag / rounding_factor

    return hpc(x_real, x_imag)


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
def is_power_of_2(N):
    """Checks if an input N is a power of 2.

    Args:
        N (int): The int in question

    Returns:
        bool: True or False
    """    
    return (N & (N - 1) == 0) and N != 0


@memoize
def sin(x):
    """Return the sine of the decimal object x as measured in radians.
    Code from the official decimal library docs. Uses the Taylor
    series approximation for the trigonometric function.

    >>> print(sin(Decimal('0.5')))
    0.4794255386042030002732879352
    >>> print(sin(0.5))
    0.479425538604
    >>> print(sin(0.5+0j))
    (0.479425538604+0j)
    """
    x = x % (2 * decimal.Decimal('3.1415926535897932384626433832'))
    decimal.getcontext().prec += 2
    i, lasts, s, fact, num, sign = 1, 0, x, 1, x, 1
    while s != lasts:
        lasts = s
        i += 2
        fact *= i * (i-1)
        num *= x * x
        sign *= -1
        s += num / fact * sign
    decimal.getcontext().prec -= 2
    return +s


@memoize
def cos(x):
    """Return the cosine of decimal object x as measured in radians.
    Code from the official decimal library docs. Uses the Taylor
    series approximation for the trigonometric function.

    >>> print(cos(Decimal('0.5')))
    0.8775825618903727161162815826
    >>> print(cos(0.5))
    0.87758256189
    >>> print(cos(0.5+0j))
    (0.87758256189+0j)
    """
    x = x % (2 * decimal.Decimal('3.1415926535897932384626433832'))
    decimal.getcontext().prec += 2
    i, lasts, s, fact, num, sign = 0, 0, 1, 1, 1, 1
    while s != lasts:
        lasts = s
        i += 2
        fact *= i * (i-1)
        num *= x * x
        sign *= -1
        s += num / fact * sign
    decimal.getcontext().prec -= 2
    return +s


@memoize
def w(n, k, N):
    """Function that computes for the twiddle factor
    used in fourier transformations.

    Args:
        n (int): index of time-domain element
        k (int): index of freq-domain element
        N (int): Number of total elements

    Returns:
        hpc: e ^ (-2πjnk / N) in rectangular form
    """
    pi = decimal.Decimal('3.1415926535897932384626433832')
    theta = -2 * pi * n * k / N
    return hpc(cos(theta), sin(theta))


def fft_helper(signal):
    """Helper function to fft() that solves for the 
    fast fourier transform of a freq-domain signal.
    For high precision applications, input list should 
    be composed of strings or hpc objects.

    Args:
        signal (list): The list (hpc, complex, or str) that
        represents the time-domain elements of the signal to 
        be transformed.

    Returns:
        list: A list of hpc objects that represents the 
        freq-domain elements of the transformed signal.
    """

    signal_copy = signal.copy()

    while not is_power_of_2(len(signal_copy)):
        signal_copy.append(0)

    N = len(signal_copy)

    if not isinstance(signal_copy[0], hpc):
        signal_copy = [hpc(item) for item in signal_copy]

    # base cases
    if N == 2:
        return [signal_copy[0] + signal_copy[1], signal_copy[0] - signal_copy[1]]
    if N == 1:
        return [signal_copy[0]]

    # split X to even, odd1, and odd3 elements
    X_even = fft_helper([signal[index] for index in range(N) if index % 2 == 0])
    X_odd1 = fft_helper([signal[index] for index in range(N) if index != 0 and (index - 1) % 4 == 0])
    X_odd3 = fft_helper([signal[index] for index in range(N) if index != 0 and (index - 3) % 4 == 0])

    # solve for the twiddle factors to be used
    w1k = [w(1, k, N) for k in range(N)]
    w3k = [w(1, 3 * k, N) for k in range(N)]

    # compute for sum and diff odd
    sum_odd = [X_odd1[index] * w1k[index] + X_odd3[index] * w3k[index] for index in range(len(X_odd1))]
    diff_odd = [X_odd1[index] * w1k[index] - X_odd3[index] * w3k[index] for index in range(len(X_odd1))]

    # Compute for the quadrants of X
    E = len(X_even) // 2

    X0 = [X_even[index] + sum_odd[index] for index in range(E)]
    X1 = [X_even[index + E] - diff_odd[index] * 1j for index in range(E)]
    X2 = [X_even[index] - sum_odd[index] for index in range(E)]
    X3 = [X_even[index + E] + diff_odd[index] * 1j for index in range(E)]

    X = flatten_list([X0, X1, X2, X3])

    return X


def fft(signal, decimal_places=28):
    """The client function that returns the fast fourier 
    transform of a freq-domain signal. The function only 
    rounds off the elements of the transformed signal, and 
    the transformation itself is handled by fft_helper().
    For high precision applications, input list should be 
    composed of strings or hpc objects.

    Args:
        signal (list): The list (hpc, complex, or str) that
        represents the time-domain elements of the signal to 
        be transformed.

    Returns:
        list: A list of rounded off numbers as strings that represent
        the freq-domain elements of the transformed signal.

    >>> print(fft(['1.23', '4.56', '7.89', '0']))
    ['13.68', '-6.66-4.56j', '4.56', '-6.66+4.56j']
    """
    return [str(hpc_round(item, decimal_places=decimal_places)) for item in fft_helper(signal)]


def ifft_helper(signal):
    """Helper function to ifft() that solves for the inverse
    fast fourier transform of a freq-domain signal. For high 
    precision applications, input list should be composed of 
    strings or hpc objects.

    Args:
        signal (list): The list (hpc, complex, or str) that
        represents the freq-domain elements of the signal to 
        be transformed.

    Returns:
        list: A list of hpc objects that represents the 
        time-domain elements of the transformed signal.
    """

    signal_copy = signal.copy()

    while not is_power_of_2(len(signal_copy)):
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


def ifft(signal, decimal_places=28):
    """The client function that returns the inverse
    fast fourier transform of a freq-domain signal. The 
    function only rounds off the elements of the transformed 
    signal, and the transformation itself is handled by 
    ifft_helper(). For high precision applications, input
    list should be composed of strings or hpc objects.

    Args:
        signal (list): The list (hpc, complex, or str) that
        represents the freq-domain elements of the signal to 
        be transformed.
        decimal_places (int, optional): The number of decimal places
        desired. Defaults to 28.

    Returns:
        list: A list of rounded off numbers as strings that represent
        the time-domain elements of the transformed signal.

    >>> print(ifft(['13.68', '-6.66-4.56j', '4.56', '-6.66+4.56j']))
    ['1.23', '4.56', '7.89', '0']
    """
    return [str(hpc_round(item, decimal_places=decimal_places)) for item in ifft_helper(signal)]


def main():
    """The main function that handles the reading and writing
    to stdin and stdout. Before printing the inverse transformed
    elements of each time signal, it rounds off the elements to 1 
    decimal place then converts it into integers.
    """    
    lines = []

    line = sys.stdin.readline()
    while line:
        lines.append(line)
        line = sys.stdin.readline()

    signals_num = int(lines[0])
    lines.pop(0)

    answers = [signals_num]

    for index in range(signals_num):

        signal = lines[index]
        
        # pop out the time signal length
        time_signal_length = re.search(r'\d+', signal).group()
        signal = signal.replace(time_signal_length, '', 1)

        # pop out the freq signal length
        freq_signal_length = re.search(r'\d+', signal).group()
        signal = signal.replace(freq_signal_length, '', 1)
        
        # convert freq elements string into a list of complex numbers as strings
        signal = signal.replace(' ', '')
        signal_regex = re.compile(r'[\+-][\d]+\.[\d]+[\+-][\d]+\.[\d]+j')

        signal = [hpc(item) for item in signal_regex.findall(signal)]

        # floor into integers the rounded off elements of the transformed signal
        time_signal = [int(hpc(item).re()) for item in ifft(signal, decimal_places=6)]
        ans = [time_signal_length]

        for index in range(int(time_signal_length)):
            ans.append(str(time_signal[index]))

        answers.append(' '.join(ans))

    for ans in answers:
        print(ans)


if __name__ == '__main__':
    main()

    # example of high precision transform
    # test = ['1.000000123456700000000001', '7.654321000000000000098765', '1.001001000000000000011234', '0.000000000000000000012345']
    # print(ifft(fft(test)))

