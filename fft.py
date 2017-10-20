import numpy as np
import math
import sys

# def dft(x):
#     """ simple fourier transform -> naive evaluation [O(N^2)] """
#     N = len(x)
#     ans = np.zeros((1, N), dtype=complex)
#     for k in range(N):
#         acu = 0
#         for n in range(N):
#             acu += x[n] * (np.power(math.e,  (- 1j * 2 * math.pi * k * n / N)))
#         ans[0, k] = acu
#     return ans


def dft(x):
    """ simple fourier transform -> naive evaluation [O(N^2)] """
    N = len(x)
    n = np.arange(N)
    npk = np.outer(n, n)
    factor = np.power(math.e, (- 1j * 2 * math.pi * npk / N))
    return np.dot(x, factor)

def fft(x, N, s):
    """ unoptimized fourier fast transform """
    ans = np.zeros(N, dtype=complex)
    if N == 1:
        ans[0] = x[0]
    else:
        ans[:N//2] = fft(x, N//2, 2*s)
        ans[N//2:] = fft(x[s:], N//2, 2*s)
        for k in range(N//2):
            t = ans[k]
            ans[k] = t + np.exp(- 1j * 2 * math.pi * k / N) * ans[k+N//2]
            ans[k+N//2] = t - np.exp(- 1j * 2 * math.pi * k / N) * ans[k+N//2]
    return ans

def fft_opt(x, N, s, index):
    """ slightly optimized fourier fast transform, using numpy """
    ans = np.zeros(N, dtype=complex)
    if N == 8:
        ans[0:8] = dft(x[index::s])
    else:
        E = fft_opt(x, N//2, 2*s, index)
        O = fft_opt(x, N//2, 2*s, index + s)
        ans[:N//2] = E + O * np.exp(- 1j * 2 * math.pi * np.arange(N//2) / N)
        ans[N//2:] = E - O * np.exp(- 1j * 2 * math.pi * np.arange(N//2) / N)
    return ans

def reverse_bit(x, bits):
    y = 0
    for i in range(bits):
        y = (y << 1) | (x & 1)
        x >>= 1
    return y

def get_bit_reverse_list(x):
    bits = len(x).bit_length() - 1
    ans = np.zeros(len(x), dtype=complex)
    for k in range(len(x)):
        ans[reverse_bit(k, bits)] = x[k]
    return ans



def fft_iter(x):
    """ unoptimized iterative fast fourier transform """
    A = get_bit_reverse_list(x)
    maxs = int(math.log(len(x), 2))
    for s in range(1, maxs+1):
        m = int(2 ** s)
        wm = np.exp(-2*np.pi*1j/m)
        for k in range(0, len(x), m):
            w = 1
            for j in range(m//2):
                t = w * A[k + j + m // 2]
                u = A[k + j]
                A[k + j] = u + t
                A[k + j + m//2] = u - t
                w = w * wm
    return A


# def fft_iter_opt(x):
#     """ iterative fast fourier transform, using numpy"""
#     A = get_bit_reverse_list(x)
#     maxs = int(math.log(len(x), 2))
#     u = np.zeros(len(x) // 2, dtype=complex)
#     t = np.zeros(len(x) // 2, dtype=complex)
#     W = np.zeros(2 ** maxs, dtype=complex)
#     ra = np.arange(2 ** maxs)
#
#     # Optimization, halves execution time by assuming real input
#     u[:len(x)//2] = A[::2]
#     A[::2] = u[:len(x)//2] + A[1::2]
#     A[1::2] = u[:len(x)//2] - A[1::2]
#
#     for s in range(2, maxs+1):
#         m = int(2 ** s)
#         W[:m//2] = np.exp(-2*np.pi*1j * ra[:m//2] / m)
#
#         for k in range(0, len(x), m):
#             t[0:m//2] = W[0:m//2] * A[k+m//2: k+m]
#             u[0:m//2] = A[k:k+m//2]
#             A[k: k + m//2] = u[0:m//2] + t[0:m//2]
#             A[k+m//2:k+m] = u[0:m//2] - t[0:m//2]
#
#     return A
def fft_iter_opt(x):
    """ iterative fast fourier transform, using numpy"""
    A = get_bit_reverse_list(x)

    maxs = int(math.log(len(x), 2))
    u = np.zeros(len(x), dtype=complex)
    t = np.zeros(len(x), dtype=complex)
    W = np.zeros(2 ** maxs, dtype=complex)
    ra = np.arange(2 ** maxs)

    # Optimization, halves execution time by assuming real input
    u[:len(x) // 2] = A[::2]
    A[::2] = u[:len(x) // 2] + A[1::2]
    A[1::2] = u[:len(x) // 2] - A[1::2]

    for s in range(2, maxs + 1):
        m = int(2 ** s)
        W[:m // 2] = np.exp(-2 * np.pi * 1j * ra[:m // 2] / m)

        for k in range(0, len(x), m):
            t[k:k+m//2] = W[0:m//2] * A[k+m//2: k+m]
            u[k:k+m//2] = A[k:k+m//2]

        if m // 2 < len(x) // m:
            for i in range(0, m//2):
                A[i::m] = u[i::m] + t[i::m]
                A[m//2+i::m] = u[i::m] - t[i::m]
        else:
            for k in range(0, len(x), m):
                A[k: k + m // 2] = u[k:k+m//2] + t[k:k+m//2]
                A[k+m//2:k+m] = u[k:k+m//2] - t[k:k+m//2]
    return A
