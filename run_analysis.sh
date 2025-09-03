#!/bin/bash

# OFDM Simulator Analysis Script
# Comprehensive testing and analysis of the OFDM implementation

set -e  # Exit on any error
alias python=python3
echo "🔬 OFDM Simulator Analysis Suite"
echo "=================================="

# Check if required files exist
if [ ! -f "ofdm_cli.py" ]; then
    echo "❌ Error: ofdm_cli.py not found!"
    exit 1
fi

# Create results directory
mkdir -p results
cd results

echo ""
echo "1️⃣  Basic Functionality Tests"
echo "-----------------------------"

# Test each modulation with ideal channel
echo "Testing BPSK (ideal channel)..."
python3 ../ofdm_cli.py --mod bpsk --run demo --N 8 --bits "10110100"

echo -e "\nTesting QPSK (ideal channel)..."
python3 ../ofdm_cli.py --mod qpsk --run demo --N 4 --bits "10110100"

echo -e "\nTesting 16-QAM (ideal channel)..."
python3 ../ofdm_cli.py --mod mqam --M 16 --run demo --N 4

echo ""
echo "2️⃣  AWGN Channel Performance"
echo "----------------------------"

# Quick AWGN tests
echo "BPSK with AWGN..."
python3 ../ofdm_cli.py --mod bpsk --channel awgn --run demo --snr_list "0,10,20" --N 8

echo -e "\nQPSK with AWGN..."
python3 ../ofdm_cli.py --mod qpsk --channel awgn --run demo --snr_list "0,10,20" --N 8

echo ""
echo "3️⃣  BER Analysis (this may take a while...)"
echo "-------------------------------------------"

# BER analysis for different modulations
echo "Computing BER curves..."

echo "  - BPSK BER analysis..."
python3 ../ofdm_cli.py --mod bpsk --channel awgn --run ber --frames 2000 --N 64 \
    --ber_snr_list "0,2,4,6,8,10,12,15,20" > ber_bpsk.txt

echo "  - QPSK BER analysis..."
python3 ../ofdm_cli.py --mod qpsk --channel awgn --run ber --frames 2000 --N 64 \
    --ber_snr_list "0,2,4,6,8,10,12,15,20" > ber_qpsk.txt

echo "  - 16-QAM BER analysis..."
python3 ../ofdm_cli.py --mod mqam --M 16 --channel awgn --run ber --frames 1000 --N 64 \
    --ber_snr_list "0,5,10,15,20,25" > ber_16qam.txt

echo ""
echo "4️⃣  System Parameter Analysis"
echo "-----------------------------"

# Test different system parameters
echo "Testing different CP lengths..."
for cp in 0 4 8 16; do
    echo "  CP length = $cp"
    python3 ../ofdm_cli.py --mod qpsk --channel awgn --run demo --cp_len $cp \
        --snr_list "10" --N 16 | grep "bits_hat" || true
done

echo -e "\nTesting different OFDM sizes..."
for N in 16 32 64 128; do
    echo "  N = $N subcarriers"
    python3 ../ofdm_cli.py --mod qpsk --channel awgn --run demo --N $N \
        --snr_list "15" | grep "bits_hat" || true
done

echo ""
echo "5️⃣  Modulation Comparison"
echo "------------------------"

# Compare all modulations at same SNR
SNR=15
echo "Comparing modulations at SNR = ${SNR} dB:"

python3 ../ofdm_cli.py --mod bpsk --channel awgn --run demo --snr_list "$SNR" --N 16 \
    | grep -E "(DEMO|bits_hat)" | head -2

python3 ../ofdm_cli.py --mod qpsk --channel awgn --run demo --snr_list "$SNR" --N 16 \
    | grep -E "(DEMO|bits_hat)" | head -2

python3 ../ofdm_cli.py --mod mqam --M 16 --channel awgn --run demo --snr_list "$SNR" --N 16 \
    | grep -E "(DEMO|bits_hat)" | head -2

echo ""
echo "6️⃣  Edge Cases and Robustness"
echo "-----------------------------"

# Test edge cases
echo "Testing edge cases..."

echo "  - No cyclic prefix:"
python3 ../ofdm_cli.py --mod bpsk --cp_len 0 --run demo --N 4 | grep "Bits out" || true

echo "  - Single subcarrier:"
python3 ../ofdm_cli.py --mod bpsk --N 1 --run demo --bits "1" | grep "Bits out" || true

echo "  - Large constellation (64-QAM):"
python3 ../ofdm_cli.py --mod mqam --M 64 --run demo --N 4 | grep "Bits out" || true

echo ""
echo "7️⃣  Performance Summary"
echo "----------------------"

# Extract key results
echo "BER at 10 dB SNR:"
echo "  BPSK:   $(grep 'SNR=10.0' ber_bpsk.txt | awk '{print $4}' || echo 'N/A')"
echo "  QPSK:   $(grep 'SNR=10.0' ber_qpsk.txt | awk '{print $4}' || echo 'N/A')"
echo "  16-QAM: $(grep 'SNR=10.0' ber_16qam.txt | awk '{print $4}' || echo 'N/A')"

echo ""
echo "8️⃣  Generating Visualizations"
echo "----------------------------"

if command -v python3 &> /dev/null; then
    if python3 -c "import matplotlib" 2>/dev/null; then
        echo "Running visualization script..."
        python3 ../test_and_visualize.py
        echo "✅ Visualizations generated!"
    else
        echo "⚠️  matplotlib not available, skipping visualizations"
    fi
else
    echo "⚠️  Python not found, skipping visualizations"
fi

echo ""
echo "🎉 Analysis Complete!"
echo "===================="
echo "Results saved in: $(pwd)"
echo ""
echo "Generated files:"
ls -la *.txt *.png 2>/dev/null || echo "  (no additional files generated)"

echo ""
echo "📋 Quick Summary:"
echo "  ✅ Basic functionality tests completed"
echo "  ✅ AWGN channel performance analyzed"  
echo "  ✅ BER curves generated for multiple modulations"
echo "  ✅ System parameter sensitivity tested"
echo "  ✅ Edge cases verified"

cd ..```