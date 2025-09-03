# OFDM Simulator Usage Guide

A comprehensive OFDM simulator supporting BPSK, QPSK, and M-QAM modulation with ideal and AWGN channels.

## Quick Start

```bash
# Basic BPSK demo with ideal channel
python ofdm_cli.py --mod bpsk --channel ideal --run demo

# QPSK with AWGN at different SNRs
python ofdm_cli.py --mod qpsk --channel awgn --run demo --snr_list "0,10,20"

# 16-QAM BER analysis
python ofdm_cli.py --mod mqam --M 16 --channel awgn --run ber --frames 5000
```

## Command Line Arguments

### Core Parameters
- `--mod {bpsk,qpsk,mqam}`: Modulation scheme (default: bpsk)
- `--M INT`: Constellation size for M-QAM (default: 16, ignored for BPSK/QPSK)
- `--channel {ideal,awgn}`: Channel type (default: ideal)
- `--N INT`: Number of subcarriers (default: 4)
- `--cp_len INT`: Cyclic prefix length (default: 1)

### Execution Mode
- `--run {demo,ber,both}`: What to execute (default: both)
  - `demo`: Single transmission example
  - `ber`: Bit error rate analysis
  - `both`: Run both demo and BER

### Demo Configuration
- `--bits STRING`: Custom bit pattern (e.g., "10110101")
- `--snr_list STRING`: SNR values for demo in dB (default: "0,10,20,30")

### BER Analysis
- `--ber_snr_list STRING`: SNR range for BER sweep (default: "0,2,4,6,8,10,12,15,20,25,30")
- `--frames INT`: Number of frames per SNR point (default: 2000)

### Other
- `--seed INT`: Random seed for reproducibility (default: 123)

## Examples

### 1. BPSK Basics
```bash
# Ideal channel demo
python ofdm_cli.py --mod bpsk --bits "1010" --N 4

# AWGN performance
python ofdm_cli.py --mod bpsk --channel awgn --snr_list "0,5,10,15,20"
```

### 2. QPSK Analysis
```bash
# Custom bit pattern (8 bits for N=4 subcarriers)
python ofdm_cli.py --mod qpsk --bits "10110100" --N 4

# BER curve
python ofdm_cli.py --mod qpsk --channel awgn --run ber --frames 10000
```

### 3. M-QAM Comparison
```bash
# 16-QAM demo
python ofdm_cli.py --mod mqam --M 16 --channel awgn --run demo

# 64-QAM BER analysis
python ofdm_cli.py --mod mqam --M 64 --channel awgn --run ber --frames 5000
```

### 4. System Parameters
```bash
# Larger OFDM system
python ofdm_cli.py --N 64 --cp_len 16 --mod qpsk --channel awgn

# No cyclic prefix
python ofdm_cli.py --cp_len 0 --mod bpsk --channel awgn
```

## Bit Requirements

The number of input bits depends on modulation and subcarriers:
- **BPSK**: N bits (1 bit per symbol)
- **QPSK**: 2×N bits (2 bits per symbol)  
- **M-QAM**: log₂(M)×N bits (log₂(M) bits per symbol)

For N=4 subcarriers:
- BPSK: 4 bits (e.g., "1010")
- QPSK: 8 bits (e.g., "10110100")
- 16-QAM: 16 bits (e.g., "1011010011001010")

## Output Interpretation

### Demo Output
- **Bits in**: Original input bits
- **X[k]**: Frequency domain symbols after modulation
- **x[n] IFFT**: Time domain samples after IFFT
- **CP added**: Signal with cyclic prefix
- **Y[k] FFT**: Received frequency domain symbols
- **Bits out**: Detected bits

### BER Output
Shows bit error rate vs SNR for performance analysis.