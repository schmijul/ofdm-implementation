import numpy as np
import argparse
from typing import Tuple, List


# ---------------------- Modulation / Demodulation ----------------------

def bpsk_map(bits: np.ndarray) -> np.ndarray:
    return np.where(bits == 0, 1.0, -1.0).astype(np.complex128)


def bpsk_demap(symbols: np.ndarray) -> np.ndarray:
    return (np.real(symbols) < 0).astype(int)


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


def map_bits(bits: np.ndarray, mod: str, M: int | None = None) -> np.ndarray:
    if mod == "bpsk":
        return bpsk_map(bits)
    elif mod == "qpsk":
        return qpsk_map(bits)
    elif mod == "mqam":
        if M is None:
            raise ValueError("M must be provided for mqam")
        return mqam_map(bits, M)
    else:
        raise ValueError(f"Unknown modulation: {mod}")


def demap_symbols(symbols: np.ndarray, mod: str, M: int | None = None) -> np.ndarray:
    if mod == "bpsk":
        return bpsk_demap(symbols)
    elif mod == "qpsk":
        return qpsk_demap(symbols)
    elif mod == "mqam":
        if M is None:
            raise ValueError("M must be provided for mqam")
        return mqam_demap(symbols, M)
    else:
        raise ValueError(f"Unknown modulation: {mod}")


# ---------------------- OFDM + Channel ----------------------

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


# ---------------------- Simulation helpers ----------------------

def ensure_bits_length(bits: np.ndarray, N: int, mod: str, M: int | None) -> None:
    if mod == "bpsk":
        need = N
    elif mod == "qpsk":
        need = 2 * N
    else:  # mqam
        if M is None:
            raise ValueError("M must be provided for mqam")
        k = int(np.log2(M))
        need = k * N
    if bits.size != need:
        raise ValueError(f"Provided bits length {bits.size} must equal {need} for {mod}")


def simulate_once(bits: np.ndarray, mod: str, N: int, cp_len: int, channel: str, snr_db: float | None, M: int | None, rng: np.random.Generator | None = None):
    X = map_bits(bits, mod, M)
    x = ofdm_modulate(X)
    tx = add_cyclic_prefix(x, cp_len)
    if channel == "ideal":
        rx = tx
    elif channel == "awgn":
        if snr_db is None:
            raise ValueError("snr_db must be provided for awgn channel")
        rx = add_awgn_time_domain(tx, snr_db, N=N, Es=1.0, rng=rng)
    else:
        raise ValueError("Unknown channel")
    rx_no_cp = remove_cyclic_prefix(rx, cp_len)
    Y = ofdm_demodulate(rx_no_cp)
    bits_hat = demap_symbols(Y, mod, M)
    return X, x, tx, Y, bits_hat


def simulate_ber(num_frames: int, N: int, mod: str, M: int | None, cp_len: int, channel: str, snr_db: float, rng: np.random.Generator | None = None) -> float:
    if rng is None:
        rng = np.random.default_rng()
    if mod == "bpsk":
        k = 1
    elif mod == "qpsk":
        k = 2
    else:
        k = int(np.log2(M))  # type: ignore[arg-type]
    err = 0
    total = 0
    for _ in range(num_frames):
        bits = rng.integers(0, 2, size=N * k, dtype=int)
        _, _, _, _, bits_hat = simulate_once(bits, mod, N, cp_len, channel, snr_db, M, rng)
        err += np.count_nonzero(bits != bits_hat)
        total += N * k
    return err / total


# ---------------------- CLI ----------------------

def parse_snr_list(s: str) -> List[float]:
    return [float(x) for x in s.split(',') if x.strip()]


def bits_from_string(bit_str: str) -> np.ndarray:
    return np.array([1 if c == '1' else 0 for c in bit_str.strip()], dtype=int)


def parse_args():
    p = argparse.ArgumentParser(description="Unified OFDM CLI: BPSK/QPSK/M-QAM (ideal/AWGN)")
    p.add_argument("--mod", choices=["bpsk", "qpsk", "mqam"], default="bpsk", help="Modulation type")
    p.add_argument("--M", type=int, default=16, help="M for M-QAM (ignored for bpsk/qpsk; qpsk corresponds to M=4)")
    p.add_argument("--channel", choices=["ideal", "awgn"], default="ideal", help="Channel type")
    p.add_argument("--run", choices=["demo", "ber", "both"], default="both", help="What to run: one-shot demo, BER sweep, or both")
    p.add_argument("--N", type=int, default=4, help="Number of subcarriers (symbols per OFDM symbol)")
    p.add_argument("--cp_len", type=int, default=1, help="Cyclic prefix length")
    p.add_argument("--bits", type=str, default=None, help="Bit string for demo (length depends on modulation)")
    p.add_argument("--snr_list", type=str, default="0,10,20,30", help="Comma-separated SNRs in dB for demo (AWGN)")
    p.add_argument("--ber_snr_list", type=str, default="0,2,4,6,8,10,12,15,20,25,30", help="Comma-separated SNRs in dB for BER sweep (AWGN)")
    p.add_argument("--frames", type=int, default=2000, help="Number of frames for BER per SNR")
    p.add_argument("--seed", type=int, default=123, help="RNG seed for reproducibility")
    return p.parse_args()


def main():
    np.set_printoptions(precision=4, suppress=True)
    args = parse_args()
    mod = args.mod
    M = args.M if mod == "mqam" else (4 if mod == "qpsk" else None)
    channel = args.channel
    N = args.N
    cp_len = args.cp_len
    rng = np.random.default_rng(args.seed)

    # Prepare bits for demo
    if args.bits is not None:
        bits_demo = bits_from_string(args.bits)
    else:
        # Generate a deterministic demo pattern
        k_demo = 1 if mod == "bpsk" else (2 if mod == "qpsk" else int(np.log2(M)))
        bits_demo = rng.integers(0, 2, size=N * k_demo, dtype=int)
    ensure_bits_length(bits_demo, N, mod, M)

    # Demo run
    if args.run in ("demo", "both"):
        if channel == "ideal":
            X, x, tx, Y, bits_hat = simulate_once(bits_demo, mod, N, cp_len, channel, None, M, rng)
            print(f"[DEMO {mod.upper()} IDEAL] N={N} CP={cp_len}")
            print("Bits in:   ", bits_demo)
            print("X[k]:      ", np.round(X, 4))
            print("x[n] IFFT: ", np.round(x, 4))
            print("CP added:  ", np.round(tx, 4))
            print("Y[k] FFT:  ", np.round(Y, 4))
            print("Bits out:  ", bits_hat)
        else:  # awgn
            for snr_db in parse_snr_list(args.snr_list):
                X, x, tx, Y, bits_hat = simulate_once(bits_demo, mod, N, cp_len, channel, snr_db, M, rng)
                print(f"[DEMO {mod.upper()} AWGN SNR={snr_db:>4.1f} dB] X: {np.round(X,4)}, Y: {np.round(Y,4)}, bits_hat: {bits_hat}")

    # BER run
    if args.run in ("ber", "both") and channel == "awgn":
        print(f"\nBER vs SNR ({mod.upper()}, N={N}, frames={args.frames}):")
        for snr_db in parse_snr_list(args.ber_snr_list):
            ber = simulate_ber(num_frames=args.frames, N=N, mod=mod, M=M, cp_len=cp_len, channel=channel, snr_db=snr_db, rng=rng)
            print(f"  SNR={snr_db:>4.1f} dB -> BER={ber:.6f}")


if __name__ == "__main__":
    main()

