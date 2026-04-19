import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate, firwin, filtfilt, butter
import os

# --------------------------
# Configuration Parameters (consistent with GNU Radio)
# --------------------------
samp_rate = 400000  # Original sample rate (USRP baseband output)
center_freq = 73.778e6  # Kept for reference, actually already down-converted
input_dir = r'F:\ljspeech'
decimation = 8  # Decimation factor
freq_offset = -49.20e3  # Residual frequency offset fine-tuning (Hz)

bandpass_low = 80  # Low cutoff frequency for speech (Hz)
bandpass_high = 7600  # High cutoff frequency for speech (Hz)
output_dir = r'F:\ljspeech\ljspeech-demod'

# Sample rate after decimation
new_samp_rate = samp_rate / decimation  # 50000 Hz

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)


# --------------------------
# 1. Read fc32 format data
# --------------------------
def read_fc32_file(file_path):
    """Read fc32 files saved by GNU Radio (complex 32-bit float)"""
    dtype = np.dtype([('real', np.float32), ('imag', np.float32)])
    data = np.fromfile(file_path, dtype=dtype)
    return data['real'] + 1j * data['imag']


# --------------------------
# 2. Pre-calculate filters (201-tap FIR, ensuring low-frequency response at 55Hz)
# --------------------------
def design_bandpass_50khz(lowcut, highcut, fs=50000, numtaps=201):
    """
    201 taps ensure steep enough low-frequency transition band (recommend >=201 taps, or use Butterworth IIR if computational resources are limited)

    Alternative (IIR, non-linear phase but faster computation):
    b, a = butter(4, [lowcut/(fs/2), highcut/(fs/2)], btype='band')
    return b, a  # Change to filtfilt(b, a, iq_baseband) when using
    """
    nyq = fs / 2
    low = lowcut / nyq
    high = highcut / nyq
    taps = firwin(numtaps, [low, high], pass_zero=False, window='hamming')
    return taps


# Pre-calculate bandpass coefficients (only once)
bp_taps = design_bandpass_50khz(bandpass_low, bandpass_high, new_samp_rate)
print(
    f"✅ Pre-calculated post-decimation bandpass filter: {len(bp_taps)}-tap FIR, passband {bandpass_low}-{bandpass_high}Hz @ {new_samp_rate / 1e3:.1f}kHz sample rate")

# --------------------------
# 3. Get file list
# --------------------------
fc32_files = [f for f in os.listdir(input_dir) if f.endswith('.fc32')]
fc32_files.sort()

print(f"Found {len(fc32_files)} fc32 files")

# --------------------------
# Process each file
# --------------------------
for fc32_file in fc32_files:
    file_path = os.path.join(input_dir, fc32_file)
    file_basename = os.path.splitext(fc32_file)[0]
    demod_raw_save_path = os.path.join(output_dir, f"{file_basename}.raw")

    print(f"\n{'=' * 60}")
    print(f"Processing: {fc32_file}")
    print(f"{'=' * 60}")

    # Read data
    print("📥 Reading IQ data...")
    iq_signal = read_fc32_file(file_path)
    if len(iq_signal) == 0:
        print(f"⚠️  Warning: No data read, skipping")
        continue

    print(f"✅ Read: {len(iq_signal)} samples, duration: {len(iq_signal) / samp_rate:.2f}s")

    # --------------------------
    # Step 1: Residual frequency offset fine-tuning (complex mixing)
    # --------------------------
    iq_signal = iq_signal - np.mean(iq_signal)  # Simple DC removal
    print(f"📶 Fine-tuning frequency offset: {freq_offset / 1e3:.2f} kHz...")
    t = np.arange(len(iq_signal)) / samp_rate
    iq_shifted = iq_signal * np.exp(-1j * 2 * np.pi * freq_offset * t)

    # --------------------------
    # Step 2: Anti-aliasing decimation
    # --------------------------
    print(f"⚡ Anti-aliasing decimation (1/{decimation}) → {new_samp_rate / 1e3:.1f}kHz...")
    # ftype='fir' ensures linear phase; zero_phase=True uses filtfilt for zero-phase implementation
    iq_baseband = decimate(iq_shifted, decimation, ftype='fir', zero_phase=True)

    print(f"✅ After decimation: {len(iq_baseband)} samples, duration: {len(iq_baseband) / new_samp_rate:.2f}s")

    # --------------------------
    # Step 3: Bandpass filtering after decimation (55-4200Hz)
    # --------------------------
    print(f"📊 Applying speech bandpass filter ({bandpass_low}-{bandpass_high}Hz)...")
    iq_filtered = filtfilt(bp_taps, 1.0, iq_baseband)

    # --------------------------
    # Step 4: AM envelope demodulation
    # --------------------------
    print("🔍 Envelope demodulation...")
    envelope = np.abs(iq_filtered)

    # --------------------------
    # Step 5: DC offset removal (performed on envelope domain to avoid I/Q phase distortion)
    # --------------------------
    print("🧹 Removing DC offset...")
    # Global mean subtraction (suitable for single speech segment)
    envelope_clean = envelope - np.mean(envelope)

    # Optional: Uncomment below if suppression of very low frequency drift (<10Hz) is needed
    # b_hp, a_hp = butter(2, 10/(new_samp_rate/2), 'high')
    # envelope_clean = filtfilt(b_hp, a_hp, envelope)

    # --------------------------
    # Step 6: Save
    # --------------------------
    envelope_clean.astype(np.float32).tofile(demod_raw_save_path)
    print(f"💾 Saved: {demod_raw_save_path}")
    print(
        f"   Length: {len(envelope_clean)} samples | Duration: {len(envelope_clean) / new_samp_rate:.2f}s | Sample rate: {new_samp_rate} Hz")

print("\n✅ All files processed successfully!")