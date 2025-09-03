import numpy as np
import argparse


def qpsk_map(bits: np.ndarray) -> np.ndarray:
    """Gray-mapped QPSK with unit symbol energy.
    Mapping (b0 b1) -> (I + jQ)/sqrt(2):
      00 -> +1 + j1
      01 -> -1 + j1
      11 -> -1 - j1
      10 -> +1 - j1
    """
    if bits.size % 2 != 0:
        raise ValueError("QPSK mapping requires an even number of bits")
    b0 = bits[0::2]
    b1 = bits[1::2]
    I = np.where(b0 == 0, 1.0, -1.0)
    Q = np.where(b1 == 0, 1.0, -1.0)
    return (I + 1j * Q) / np.sqrt(2)


def qpsk_demap(symbols: np.ndarray) -> np.ndarray:
    I = np.real(symbols)
    Q = np.imag(symbols)
    b0_hat = (I < 0).astype(int)
    b1_hat = (Q < 0).astype(int)
    out = np.empty(2 * symbols.size, dtype=int)
    out[0::2] = b0_hat
    out[1::2] = b1_hat
    return out


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
    p = argparse.ArgumentParser(description="Ideal QPSK-OFDM example (no noise)")
    p.add_argument("--N", type=int, default=4, help="Number of subcarriers (QPSK symbols per OFDM symbol)")
    p.add_argument("--cp_len", type=int, default=1, help="Cyclic prefix length")
    p.add_argument("--bits", type=str, default="00011110", help="Bit string to transmit (length must equal 2*N)")
    return p.parse_args()


def bits_from_string(bit_str: str) -> np.ndarray:
    return np.array([1 if c == '1' else 0 for c in bit_str.strip()], dtype=int)


def main():
    np.set_printoptions(precision=4, suppress=True)
    args = parse_args()
    N = args.N
    cp_len = args.cp_len
    bits = bits_from_string(args.bits)
    if bits.size != 2 * N:
        raise ValueError(f"Provided bits length {bits.size} must equal 2*N={2*N}")

    X = qpsk_map(bits)
    x = ofdm_modulate(X)
    tx = add_cyclic_prefix(x, cp_len)
    rx = tx.copy()  # Ideal channel
    rx_no_cp = remove_cyclic_prefix(rx, cp_len)
    Y = ofdm_demodulate(rx_no_cp)
    bits_hat = qpsk_demap(Y)

    print("[QPSK IDEAL] Input bits:     ", bits)
    print("[QPSK IDEAL] QPSK symbols X: ", np.round(X, 4))
    print("[QPSK IDEAL] IFFT x[n]:      ", np.round(x, 4))
    print(f"[QPSK IDEAL] With CP (L={cp_len}):", np.round(tx, 4))
    print("[QPSK IDEAL] FFT Y[k]:       ", np.round(Y, 4))
    print("[QPSK IDEAL] Detected bits:  ", bits_hat)


if __name__ == "__main__":
    main()

