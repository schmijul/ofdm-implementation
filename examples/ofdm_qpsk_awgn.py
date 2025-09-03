import numpy as np
import argparse


def qpsk_map(bits: np.ndarray) -> np.ndarray:
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


def add_awgn_time_domain(x: np.ndarray, snr_db: float, N: int, Es: float = 1.0, rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    snr_lin = 10 ** (snr_db / 10.0)
    N0 = Es / (N * snr_lin)
    sigma2_dim = N0 / 2.0
    noise = rng.normal(0.0, np.sqrt(sigma2_dim), size=x.shape) + 1j * rng.normal(0.0, np.sqrt(sigma2_dim), size=x.shape)
    return x + noise


def simulate_once(bits: np.ndarray, cp_len: int, snr_db: float, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = bits.size // 2
    X = qpsk_map(bits)
    x = ofdm_modulate(X)
    tx = add_cyclic_prefix(x, cp_len)
    rx = add_awgn_time_domain(tx, snr_db, N=N, Es=1.0, rng=rng)
    rx_no_cp = remove_cyclic_prefix(rx, cp_len)
    Y = ofdm_demodulate(rx_no_cp)
    bits_hat = qpsk_demap(Y)
    return X, Y, bits_hat


def simulate_ber(num_frames: int, N: int, cp_len: int, snr_db: float, rng: np.random.Generator | None = None) -> float:
    if rng is None:
        rng = np.random.default_rng()
    err = 0
    total = 0
    for _ in range(num_frames):
        bits = rng.integers(0, 2, size=2 * N, dtype=int)
        _, _, bits_hat = simulate_once(bits, cp_len, snr_db, rng=rng)
        err += np.count_nonzero(bits != bits_hat)
        total += 2 * N
    return err / total


def parse_args():
    p = argparse.ArgumentParser(description="QPSK-OFDM over AWGN with SNR sweep")
    p.add_argument("--N", type=int, default=4, help="Number of subcarriers (QPSK symbols per OFDM symbol)")
    p.add_argument("--cp_len", type=int, default=1, help="Cyclic prefix length")
    p.add_argument("--bits", type=str, default="00011110", help="Bit string to transmit (length must equal 2*N)")
    p.add_argument("--snr_list", type=str, default="0,5,10,15,20,30", help="Comma-separated SNRs in dB for demos")
    p.add_argument("--ber_snr_list", type=str, default="0,2,4,6,8,10,12,15,20,25,30", help="Comma-separated SNRs in dB for BER sweep")
    p.add_argument("--frames", type=int, default=2000, help="Number of frames for BER per SNR")
    p.add_argument("--seed", type=int, default=123, help="RNG seed for reproducibility")
    return p.parse_args()


def bits_from_string(bit_str: str) -> np.ndarray:
    return np.array([1 if c == '1' else 0 for c in bit_str.strip()], dtype=int)


def parse_snr_list(s: str) -> list[float]:
    return [float(x) for x in s.split(',') if x.strip()]


def main():
    np.set_printoptions(precision=4, suppress=True)
    args = parse_args()
    N = args.N
    cp_len = args.cp_len
    bits = bits_from_string(args.bits)
    if bits.size != 2 * N:
        raise ValueError(f"Provided bits length {bits.size} must equal 2*N={2*N}")
    demo_snrs = parse_snr_list(args.snr_list)
    ber_snrs = parse_snr_list(args.ber_snr_list)

    for snr_db in demo_snrs:
        X, Y, bits_hat = simulate_once(bits, cp_len, snr_db, rng=np.random.default_rng(args.seed))
        print(f"[QPSK AWGN SNR={snr_db:>4.1f} dB] X: {np.round(X,4)}, Y: {np.round(Y,4)}, bits_hat: {bits_hat}")

    print(f"\nQPSK BER vs SNR (random bits, N={N}, frames={args.frames}):")
    rng = np.random.default_rng(args.seed)
    for snr_db in ber_snrs:
        ber = simulate_ber(num_frames=args.frames, N=N, cp_len=cp_len, snr_db=snr_db, rng=rng)
        print(f"  SNR={snr_db:>4.1f} dB -> BER={ber:.6f}")


if __name__ == "__main__":
    main()

