import numpy as np


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


def main():
    # Setup
    N = 4
    bits = np.array([1, 0, 1, 0], dtype=int)

    # Step 1: BPSK mapping (0->+1, 1->-1)
    X = bpsk_map(bits)

    # Step 2: OFDM modulation (IFFT)
    x = ofdm_modulate(X)

    # Step 3: Add cyclic prefix (e.g., L=1)
    cp_len = 1
    tx = add_cyclic_prefix(x, cp_len)

    # Channel (identity / no noise for this example)
    rx = tx.copy()

    # Step 4: Remove CP and demodulate (FFT)
    rx_no_cp = remove_cyclic_prefix(rx, cp_len)
    Y = ofdm_demodulate(rx_no_cp)

    # Decisions
    bits_hat = bpsk_demap(Y)

    # Pretty-print results
    np.set_printoptions(precision=4, suppress=True)
    print("Input bits:", bits)
    print("BPSK symbols X[k]:", X)
    print("IFFT time x[n]:    ", x)
    print(f"With CP (L={cp_len}):   ", tx)
    print("After FFT Y[k]:    ", Y)
    print("Detected bits:     ", bits_hat)

    # Validate against the manual example values
    # Expected x: [0, 0, -1, 0]
    expected_x = np.array([0, 0, -1, 0], dtype=np.complex128)
    if np.allclose(x, expected_x, atol=1e-12):
        print("Matches expected time-domain samples.")
    else:
        print("Note: time-domain samples differ from the manual example.")


if __name__ == "__main__":
    main()

