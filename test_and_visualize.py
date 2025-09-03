#!/usr/bin/env python3
"""
Test suite and visualization for OFDM simulator.
Generates plots for constellation diagrams, BER curves, and spectral analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys
from pathlib import Path
import json
from typing import Dict, List, Tuple, Any

# Import the OFDM simulator functions
try:
    from ofdm_cli import (
        simulate_once, simulate_ber, map_bits, ofdm_modulate, 
        add_cyclic_prefix, add_awgn_time_domain
    )
except ImportError:
    print("Error: Cannot import ofdm_cli.py. Make sure it's in the same directory.")
    sys.exit(1)


class OFDMTester:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.results: Dict[str, Any] = {}
        
    def test_basic_functionality(self) -> bool:
        """Test basic OFDM functionality with all modulations."""
        print("Testing basic functionality...")
        
        tests = [
            ("bpsk", None, np.array([1, 0, 1, 0])),
            ("qpsk", None, np.array([1, 0, 1, 0, 0, 1, 0, 1])),
            ("mqam", 16, self.rng.integers(0, 2, 16))
        ]
        
        all_passed = True
        for mod, M, bits in tests:
            try:
                X, x, tx, Y, bits_hat = simulate_once(
                    bits, mod, N=4, cp_len=1, channel="ideal", 
                    snr_db=None, M=M, rng=self.rng
                )
                
                # Check perfect reconstruction
                if not np.array_equal(bits, bits_hat):
                    print(f"❌ {mod.upper()} failed: bits don't match")
                    all_passed = False
                else:
                    print(f"✅ {mod.upper()} passed")
                    
            except Exception as e:
                print(f"❌ {mod.upper()} failed with error: {e}")
                all_passed = False
                
        return all_passed
    
    def generate_ber_data(self, snr_range: np.ndarray, frames: int = 1000) -> Dict[str, np.ndarray]:
        """Generate BER data for different modulations."""
        print(f"Generating BER data ({frames} frames per SNR)...")
        
        modulations = [
            ("BPSK", "bpsk", None),
            ("QPSK", "qpsk", None), 
            ("16-QAM", "mqam", 16),
            ("64-QAM", "mqam", 64)
        ]
        
        ber_data = {}
        
        for name, mod, M in modulations:
            print(f"  Computing {name}...")
            ber_values = []
            
            for snr_db in snr_range:
                ber = simulate_ber(
                    num_frames=frames, N=64, mod=mod, M=M, 
                    cp_len=16, channel="awgn", snr_db=snr_db, rng=self.rng
                )
                ber_values.append(ber)
                
            ber_data[name] = np.array(ber_values)
            
        return ber_data
    
    def plot_constellations(self, N: int = 64, snr_db: float = 20.0):
        """Plot constellation diagrams for different modulations."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Constellation Diagrams (N={N}, SNR={snr_db} dB)', fontsize=14)
        
        modulations = [
            ("BPSK", "bpsk", None),
            ("QPSK", "qpsk", None),
            ("16-QAM", "mqam", 16),
            ("64-QAM", "mqam", 64),
            ("BPSK Noisy", "bpsk", None),
            ("16-QAM Noisy", "mqam", 16)
        ]
        
        for idx, (name, mod, M) in enumerate(modulations):
            ax = axes[idx // 3, idx % 3]
            
            # Generate random bits
            if mod == "bpsk":
                k = 1
            elif mod == "qpsk": 
                k = 2
            else:
                k = int(np.log2(M))
                
            bits = self.rng.integers(0, 2, N * k)
            
            # Add noise for last two plots
            add_noise = idx >= 4
            channel = "awgn" if add_noise else "ideal"
            snr = snr_db if add_noise else None
            
            _, _, _, Y, _ = simulate_once(
                bits, mod, N, cp_len=16, channel=channel, 
                snr_db=snr, M=M, rng=self.rng
            )
            
            # Plot constellation
            ax.scatter(np.real(Y), np.imag(Y), alpha=0.6, s=20)
            ax.set_xlabel('In-phase')
            ax.set_ylabel('Quadrature')
            ax.set_title(name)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
        plt.tight_layout()
        plt.savefig('constellation_diagrams.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_ber_curves(self, snr_range: np.ndarray, ber_data: Dict[str, np.ndarray]):
        """Plot BER vs SNR curves."""
        plt.figure(figsize=(10, 6))
        
        colors = ['b-o', 'r-s', 'g-^', 'm-d']
        
        for (name, ber_values), color in zip(ber_data.items(), colors):
            plt.semilogy(snr_range, ber_values, color, label=name, markersize=4)
            
        plt.xlabel('SNR (dB)')
        plt.ylabel('Bit Error Rate')
        plt.title('BER Performance Comparison (OFDM, N=64, CP=16)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(snr_range[0], snr_range[-1])
        plt.ylim(1e-5, 1)
        
        plt.tight_layout()
        plt.savefig('ber_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_spectral_analysis(self, N: int = 64):
        """Plot time and frequency domain signals."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'OFDM Signal Analysis (N={N})', fontsize=14)
        
        # Generate QPSK signal
        bits = self.rng.integers(0, 2, 2 * N)
        X, x, tx, _, _ = simulate_once(
            bits, "qpsk", N, cp_len=16, channel="ideal", 
            snr_db=None, M=None, rng=self.rng
        )
        
        # Time domain - original
        axes[0,0].plot(np.real(x), 'b-', label='Real')
        axes[0,0].plot(np.imag(x), 'r-', label='Imag')
        axes[0,0].set_title('Time Domain (after IFFT)')
        axes[0,0].set_xlabel('Sample n')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Time domain - with CP
        axes[0,1].plot(np.real(tx), 'b-', label='Real')
        axes[0,1].plot(np.imag(tx), 'r-', label='Imag')
        axes[0,1].set_title('Time Domain (with CP)')
        axes[0,1].set_xlabel('Sample n')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Frequency domain - magnitude
        freqs = np.arange(N) - N//2
        X_shifted = np.fft.fftshift(X)
        axes[1,0].stem(freqs, np.abs(X_shifted), basefmt=' ')
        axes[1,0].set_title('Frequency Domain (Magnitude)')
        axes[1,0].set_xlabel('Subcarrier k')
        axes[1,0].set_ylabel('|X[k]|')
        axes[1,0].grid(True, alpha=0.3)
        
        # Power spectral density
        psd = np.abs(np.fft.fft(x, 512))**2
        freqs_psd = np.linspace(-0.5, 0.5, 512)
        axes[1,1].plot(freqs_psd, 10*np.log10(np.fft.fftshift(psd)))
        axes[1,1].set_title('Power Spectral Density')
        axes[1,1].set_xlabel('Normalized Frequency')
        axes[1,1].set_ylabel('PSD (dB)')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('spectral_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_cli_tests(self) -> bool:
        """Test CLI functionality."""
        print("Testing CLI interface...")
        
        test_commands = [
            ["python3", "ofdm_cli.py", "--mod", "bpsk", "--run", "demo", "--N", "4"],
            ["python3", "ofdm_cli.py", "--mod", "qpsk", "--channel", "awgn", "--run", "demo", "--snr_list", "10"],
            ["python3", "ofdm_cli.py", "--mod", "mqam", "--M", "16", "--run", "demo", "--N", "4"]
        ]
        
        all_passed = True
        for cmd in test_commands:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode != 0:
                    print(f"❌ CLI test failed: {' '.join(cmd[2:])}")
                    print(f"   Error: {result.stderr}")
                    all_passed = False
                else:
                    print(f"✅ CLI test passed: {' '.join(cmd[2:])}")
            except subprocess.TimeoutExpired:
                print(f"❌ CLI test timed out: {' '.join(cmd[2:])}")
                all_passed = False
            except Exception as e:
                print(f"❌ CLI test error: {e}")
                all_passed = False
                
        return all_passed
    
    def save_results(self, filename: str = "test_results.json"):
        """Save test results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Results saved to {filename}")


def main():
    print("🚀 OFDM Simulator Test Suite")
    print("=" * 50)
    
    tester = OFDMTester()
    
    # Run basic functionality tests
    basic_ok = tester.test_basic_functionality()
    print()
    
    # Run CLI tests
    cli_ok = tester.run_cli_tests()
    print()
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # Constellation diagrams
    tester.plot_constellations()
    
    # BER analysis
    snr_range = np.arange(0, 21, 2)
    ber_data = tester.generate_ber_data(snr_range, frames=500)  # Reduced for speed
    tester.plot_ber_curves(snr_range, ber_data)
    
    # Spectral analysis
    tester.plot_spectral_analysis()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"  Basic functionality: {'✅ PASS' if basic_ok else '❌ FAIL'}")
    print(f"  CLI interface: {'✅ PASS' if cli_ok else '❌ FAIL'}")
    print("  Visualizations: ✅ Generated")
    print("\nGenerated files:")
    print("  - constellation_diagrams.png")
    print("  - ber_curves.png") 
    print("  - spectral_analysis.png")
    
    # Save results
    tester.results = {
        "basic_tests": basic_ok,
        "cli_tests": cli_ok,
        "snr_range": snr_range.tolist(),
        "ber_data": {k: v.tolist() for k, v in ber_data.items()}
    }
    tester.save_results()


if __name__ == "__main__":
    main()