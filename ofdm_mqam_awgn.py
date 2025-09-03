import numpy as np
import argparse


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
    return (2 * np.arange(m) - (m - 1)).astype(float)


def mqam_map(bits: np.ndarray, M: int) -> np.ndarray:
    if (M & (M - 1)) != 0:
        raise ValueError("M must be a power of two (square M-QAM)")
    sqrtM = int(np.sqrt(M))
    if sqrtM * sqrtM != M:
        raise ValueError("M must be a perfect square (e.g., 4,16,64)")
    p = int(np.log2(sqrtM))
    k = 2 * p
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

    def axis_bits(vals: np.ndarray) -> np.ndarray:
        vals = vals * norm
        idx = np.rint((vals - L[0]) / 2.0).astype(int)
        idx = np.clip(idx, 0, sqrtM - 1)
        out = np.empty((vals.size, p), dtype=int)
        for i, g in enumerate(idx):
            n = gray_decode(int(g))
            out[i] = bits_from_int(n, p)
        return out

    I_bits = axis_bits(np.real(symbols))
    Q_bits = axis_bits(np.imag(symbols))

    out_bits = []
    for i in range(symbols.size):
        out_bits.append(I_bits[i])
        out_bits.append(Q_bits[i])
    return np.concatenate(out_bits)


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


def simulate_once(bits: np.ndarray, M: int, cp_len: int, snr_db: float, rng: np.random.Generator | None = None):
    N = bits.size // int(np.log2(M))
    X = mqam_map(bits, M)
    x = ofdm_modulate(X)
    tx = add_cyclic_prefix(x, cp_len)
    rx = add_awgn_time_domain(tx, snr_db, N=N, Es=1.0, rng=rng)
    rx_no_cp = remove_cyclic_prefix(rx, cp_len)
    Y = ofdm_demodulate(rx_no_cp)
    bits_hat = mqam_demap(Y, M)
    return X, Y, bits_hat


def simulate_ber(num_frames: int, N: int, M: int, cp_len: int, snr_db: float, rng: np.random.Generator | None = None) -> float:
    if rng is None:
        rng = np.random.default_rng()
    k = int(np.log2(M))
    err = 0
    total = 0
    for _ in range(num_frames):
        bits = rng.integers(0, 2, size=N * k, dtype=int)
        _, _, bits_hat = simulate_once(bits, M, cp_len, snr_db, rng=rng)
        err += np.count_nonzero(bits != bits_hat)
        total += N * k
    return err / total


def parse_args():
    p = argparse.ArgumentParser(description="M-QAM OFDM over AWGN (Gray mapping) with SNR sweep")
    p.add_argument("--M", type=int, default=16, help="Constellation size (square M-QAM: 16,64,...)")
    p.add_argument("--N", type=int, default=4, help="Number of subcarriers (M-QAM symbols per OFDM symbol)")
    p.add_argument("--cp_len", type=int, default=1, help="Cyclic prefix length")
    p.add_argument("--bits", type=str, default="0001111000011110", help="Bit string (length must equal N*log2(M))")
    p.add_argument("--snr_list", type=str, default="0,10,20,30", help="Comma-separated SNRs in dB for demos")
    p.add_argument("--ber_snr_list", type=str, default="0,6,12,18,24,30", help="Comma-separated SNRs in dB for BER sweep")
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
    M = args.M
    N = args.N
    cp_len = args.cp_len
    bits = bits_from_string(args.bits)
    k = int(np.log2(M))
    if bits.size != k * N:
        raise ValueError(f"Provided bits length {bits.size} must equal N*log2(M)={k*N}")

    demo_snrs = parse_snr_list(args.snr_list)
    ber_snrs = parse_snr_list(args.ber_snr_list)

    for snr_db in demo_snrs:
        X, Y, bits_hat = simulate_once(bits, M, cp_len, snr_db, rng=np.random.default_rng(args.seed))
        print(f"[M-QAM AWGN M={M} SNR={snr_db:>4.1f} dB] X: {np.round(X,4)}, Y: {np.round(Y,4)}, bits_hat: {bits_hat}")

    print(f"\nM-QAM BER vs SNR (random bits, M={M}, N={N}, frames={args.frames}):")
    rng = np.random.default_rng(args.seed)
    for snr_db in ber_snrs:
        ber = simulate_ber(num_frames=args.frames, N=N, M=M, cp_len=cp_len, snr_db=snr_db, rng=rng)
        print(f"  SNR={snr_db:>4.1f} dB -> BER={ber:.6f}")


if __name__ == "__main__":
    main()

