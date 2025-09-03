import numpy as np
import argparse


def bpsk_map(bits: np.ndarray) -> np.ndarray:
    """Map bits {0,1} to BPSK symbols {+1,-1} (0->+1, 1->-1)."""
    return np.where(bits == 0, 1.0, -1.0).astype(np.complex128)


def add_cyclic_prefix(x: np.ndarray, cp_len: int) -> np.ndarray:
    if cp_len <= 0:
        return x.copy()
    return np.concatenate([x[-cp_len:], x])


def remove_cyclic_prefix(y: np.ndarray, cp_len: int) -> np.ndarray:
    if cp_len <= 0:
        return y.copy()
    return y[cp_len:]


def ofdm_modulate(symbols: np.ndarray) -> np.ndarray:
    """OFDM modulation via IFFT (NumPy ifft includes 1/N normalization)."""
    return np.fft.ifft(symbols)


def ofdm_demodulate(samples: np.ndarray) -> np.ndarray:
    """OFDM demodulation via FFT (inverse of NumPy ifft)."""
    return np.fft.fft(samples)


def bpsk_demap(symbols: np.ndarray) -> np.ndarray:
    """Demap BPSK symbols back to bits using real-part decision."""
    return (np.real(symbols) < 0).astype(int)


def parse_args():
    p = argparse.ArgumentParser(description="Ideal BPSK-OFDM example (no noise)")
    p.add_argument("--N", type=int, default=4, help="Number of subcarriers (bits per OFDM symbol)")
    p.add_argument("--cp_len", type=int, default=1, help="Cyclic prefix length")
    p.add_argument("--bits", type=str, default="1010", help="Bit string to transmit (length must equal N)")
    return p.parse_args()


def bits_from_string(bit_str: str) -> np.ndarray:
    arr = np.array([1 if c == '1' else 0 for c in bit_str.strip()], dtype=int)
    return arr


def main():
    # Ideal OFDM channel example (no noise, no distortion)
    args = parse_args()
    N = args.N
    cp_len = args.cp_len
    bits = bits_from_string(args.bits)
    if bits.size != N:
        raise ValueError(f"Provided bits length {bits.size} must equal N={N}")

    X = bpsk_map(bits)
    x = ofdm_modulate(X)
    tx = add_cyclic_prefix(x, cp_len)
    rx = tx.copy()  # Ideal channel
    rx_no_cp = remove_cyclic_prefix(rx, cp_len)
    Y = ofdm_demodulate(rx_no_cp)
    bits_hat = bpsk_demap(Y)

    np.set_printoptions(precision=4, suppress=True)
    print("[IDEAL] Input bits:     ", bits)
    print("[IDEAL] BPSK symbols X: ", X)
    print("[IDEAL] IFFT x[n]:      ", x)
    print(f"[IDEAL] With CP (L={cp_len}):", tx)
    print("[IDEAL] FFT Y[k]:       ", Y)
    print("[IDEAL] Detected bits:  ", bits_hat)

    expected_x = np.array([0, 0, -1, 0], dtype=np.complex128)
    if np.allclose(x, expected_x, atol=1e-12):
        print("[IDEAL] Matches expected time-domain samples.")
    else:
        print("[IDEAL] Note: time-domain samples differ from manual example.")


if __name__ == "__main__":
    main()
