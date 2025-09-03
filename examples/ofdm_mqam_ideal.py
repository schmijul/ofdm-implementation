import numpy as np
import argparse
from typing import Tuple


def gray_encode(n: int) -> int:
    return n ^ (n >> 1)


def gray_decode(g: int) -> int:
    b = 0
    while g:
        b ^= g
        g >>= 1
    return b


def int_from_bits(bits: np.ndarray) -> int:
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def bits_from_int(v: int, width: int) -> np.ndarray:
    out = np.zeros(width, dtype=int)
    for i in range(width - 1, -1, -1):
        out[i] = v & 1
        v >>= 1
    return out


def pam_levels(m: int) -> np.ndarray:
    # m must be power of 2 (sqrt(M))
    return (2 * np.arange(m) - (m - 1)).astype(float)


def mqam_map(bits: np.ndarray, M: int) -> np.ndarray:
    if (M & (M - 1)) != 0:
        raise ValueError("M must be a power of two (square M-QAM)")
    sqrtM = int(np.sqrt(M))
    if sqrtM * sqrtM != M:
        raise ValueError("M must be a perfect square (e.g., 4,16,64)")
    p = int(np.log2(sqrtM))  # bits per I/Q axis
    k = 2 * p  # total bits per symbol
    if bits.size % k != 0:
        raise ValueError(f"Number of bits must be a multiple of log2(M)={k}")

    L = pam_levels(sqrtM)
    norm = np.sqrt((2.0 / 3.0) * (M - 1))

    num_syms = bits.size // k
    syms = np.empty(num_syms, dtype=np.complex128)
    for i in range(num_syms):
        bI = bits[i * k : i * k + p]
        bQ = bits[i * k + p : i * k + 2 * p]
        nI = int_from_bits(bI)
        nQ = int_from_bits(bQ)
        gI = gray_encode(nI)
        gQ = gray_encode(nQ)
        aI = L[gI]
        aQ = L[gQ]
        syms[i] = (aI + 1j * aQ) / norm
    return syms


def mqam_demap(symbols: np.ndarray, M: int) -> np.ndarray:
    sqrtM = int(np.sqrt(M))
    p = int(np.log2(sqrtM))
    k = 2 * p
    L = pam_levels(sqrtM)
    norm = np.sqrt((2.0 / 3.0) * (M - 1))

    # Map amplitude back to Gray index by nearest neighbor
    def axis_bits(vals: np.ndarray) -> np.ndarray:
        # vals: 1D float array of either I or Q
        vals = vals * norm
        # indices in [0, sqrtM-1] corresponding to levels L
        idx = np.rint((vals - L[0]) / 2.0).astype(int)
        idx = np.clip(idx, 0, sqrtM - 1)
        # convert Gray index to binary index, then to bits
        out = np.empty((vals.size, p), dtype=int)
        for i, g in enumerate(idx):
            n = gray_decode(int(g))
            out[i] = bits_from_int(n, p)
        return out

    I_bits = axis_bits(np.real(symbols))
    Q_bits = axis_bits(np.imag(symbols))

    out2 = []
    for i in range(symbols.size):
        out2.append(I_bits[i])
        out2.append(Q_bits[i])
    return np.concatenate(out2)


def ofdm_modulate(symbols: np.ndarray) -> np.ndarray:
    return np.fft.ifft(symbols)


def ofdm_demodulate(samples: np.ndarray) -> np.ndarray:
    return np.fft.fft(samples)


def add_cyclic_prefix(x: np.ndarray, cp_len: int) -> np.ndarray:
    if cp_len <= 0:
        return x.copy()
    return np.concatenate([x[-cp_len:], x])


def remove_cyclic_prefix(y: np.ndarray, cp_len: int) -> np.ndarray:
    if cp_len <= 0:
        return y.copy()
    return y[cp_len:]


def parse_args():
    p = argparse.ArgumentParser(description="Ideal M-QAM OFDM example (no noise)")
    p.add_argument("--M", type=int, default=16, help="Constellation size (square M-QAM: 4,16,64,...) memo: QPSK is M=4")
    p.add_argument("--N", type=int, default=4, help="Number of subcarriers (M-QAM symbols per OFDM symbol)")
    p.add_argument("--cp_len", type=int, default=1, help="Cyclic prefix length")
    p.add_argument("--bits", type=str, default="0001111000011110", help="Bit string (length must equal N*log2(M))")
    return p.parse_args()


def bits_from_string(bit_str: str) -> np.ndarray:
    return np.array([1 if c == '1' else 0 for c in bit_str.strip()], dtype=int)


def main():
    np.set_printoptions(precision=4, suppress=True)
    args = parse_args()
    M = args.M
    N = args.N
    cp_len = args.cp_len
    bits = bits_from_string(args.bits)
    k = int(np.log2(M))
    if bits.size != k * N:
        raise ValueError(f"Provided bits length {bits.size} must equal N*log2(M)={k*N}")

    X = mqam_map(bits, M)
    x = ofdm_modulate(X)
    tx = add_cyclic_prefix(x, cp_len)
    rx = tx.copy()
    rx_no_cp = remove_cyclic_prefix(rx, cp_len)
    Y = ofdm_demodulate(rx_no_cp)
    bits_hat = mqam_demap(Y, M)

    print(f"[M-QAM IDEAL] M={M} N={N} CP={cp_len}")
    print("[M-QAM IDEAL] Input bits:     ", bits)
    print("[M-QAM IDEAL] MQAM symbols X: ", np.round(X, 4))
    print("[M-QAM IDEAL] IFFT x[n]:      ", np.round(x, 4))
    print("[M-QAM IDEAL] FFT Y[k]:       ", np.round(Y, 4))
    print("[M-QAM IDEAL] Detected bits:  ", bits_hat)


if __name__ == "__main__":
    main()
