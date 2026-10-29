"""
COSMOS Virtual Hardware Generator (Digital Twin)
================================================
Emulates a 2-channel frontal EEG headband (F3 & F4 nodes) with:
- Biological brain rhythms: Alpha (8-13 Hz), Delta (0.5-4 Hz), Beta (13-30 Hz)
- 1/f Pink Noise characteristic of human brain tissue
- AD8232 Analog Front-End (AFE) bandpass filtering (0.5 Hz - 40 Hz)
- ATmega328P 10-bit ADC quantization (0-1023 integer counts, 3.3V reference)
- Real-time physiological state toggling: NORMAL vs ISCHEMIC_STROKE
- Physiological artifact injection: EOG (Eye Blink) & EMG (Jaw Clench)

Outputs clean, structured packets ready to be streamed over serial or HTTP/WebSocket.
"""

import time
import math
import random
import numpy as np
from scipy.signal import butter, lfilter


class CosmosVirtualHardware:
    def __init__(self, sampling_rate: int = 256, v_ref: float = 3.3):
        """
        Initializes the Virtual Hardware Digital Twin.
        
        :param sampling_rate: Sampling frequency in Hz (standard Arduino ADC loop rate = 256 Hz)
        :param v_ref: Reference voltage in Volts (3.3V for low-noise biopotential setups)
        """
        self.fs = sampling_rate
        self.v_ref = v_ref
        self.adc_resolution = 1024  # 10-bit ADC has 2^10 = 1024 discrete levels (0 to 1023)
        self.time_cursor = 0.0
        
        # State management: "NORMAL", "STROKE_LEFT", "STROKE_RIGHT", "STRESS"
        self.patient_state = "NORMAL"
        self.inject_blink = False
        self.inject_clench = False
        
        # Simulate AD8232 Hardware Analog Bandpass Filter (0.5 Hz highpass, 40 Hz lowpass)
        nyquist = 0.5 * self.fs
        low_cutoff = 0.5 / nyquist
        high_cutoff = 40.0 / nyquist
        self.filter_b, self.filter_a = butter(2, [low_cutoff, high_cutoff], btype='band')

    def set_state(self, new_state: str):
        """Switches the simulated biological condition."""
        valid_states = ["NORMAL", "STROKE_LEFT", "STROKE_RIGHT", "STRESS"]
        if new_state.upper() in valid_states:
            self.patient_state = new_state.upper()
            print(f"\n[HARDWARE EVENT] >> Patient State switched to: {self.patient_state}")
        else:
            print(f"[ERROR] Unknown state '{new_state}'. Valid options: {valid_states}")

    def trigger_eye_blink(self):
        """Simulates an Electrooculographic (EOG) eye blink artifact (+150 uV slow deflection)."""
        self.inject_blink = True
        print("[HARDWARE EVENT] >> Injected transient: Eye Blink (EOG)")

    def trigger_jaw_clench(self):
        """Simulates an Electromyographic (EMG) muscle artifact (high-frequency burst)."""
        self.inject_clench = True
        print("[HARDWARE EVENT] >> Injected transient: Jaw Clench (EMG)")

    def _generate_pink_noise(self, n_samples: int) -> np.ndarray:
        """
        Generates 1/f Pink Noise characteristic of biological cerebral backgrounds.
        """
        white_noise = np.random.randn(n_samples)
        fft_vals = np.fft.rfft(white_noise)
        freqs = np.fft.rfftfreq(n_samples)
        freqs[0] = 1.0  # Avoid zero division at DC
        scale = 1.0 / np.sqrt(freqs)
        scale[0] = 0.0  # Set DC component to zero
        pink_fft = fft_vals * scale
        pink_time = np.fft.irfft(pink_fft, n=n_samples)
        return pink_time * 6.0  # Scale background noise to approx ~6 uV RMS

    def read_packet(self, duration_sec: float = 0.25) -> dict:
        """
        Reads one time-frame of data from the virtual headband.
        Default 0.25 seconds @ 256 Hz = 64 biopotential samples per packet.
        
        Returns:
            dict containing raw microvolt traces, 10-bit quantized ADC counts,
            timestamps, and simulated battery telemetry.
        """
        n_samples = int(self.fs * duration_sec)
        t = np.linspace(self.time_cursor, self.time_cursor + duration_sec, n_samples, endpoint=False)
        self.time_cursor += duration_sec

        # 1. BIOLOGICAL WAVE GENERATION (Microvolts)
        # -------------------------------------------------------------
        # Base amplitudes for normal resting state
        alpha_amp_f3, alpha_amp_f4 = 25.0, 25.0  # Normal 10 Hz Alpha: ~25 uV
        delta_amp_f3, delta_amp_f4 = 7.0, 7.0    # Normal 2 Hz Delta: ~7 uV
        beta_amp_f3, beta_amp_f4   = 5.0, 5.0    # Normal 20 Hz Beta: ~5 uV

        if self.patient_state == "STROKE_LEFT":
            # Acute Left MCA Ischemic Stroke:
            # - Left hemisphere (F3): Alpha wave collapses (-80%), Delta wave surges (+400%)
            # - Right hemisphere (F4): Remains intact
            alpha_amp_f3 = 4.0
            delta_amp_f3 = 45.0
            alpha_amp_f4 = 25.0
            delta_amp_f4 = 7.0

        elif self.patient_state == "STROKE_RIGHT":
            # Acute Right MCA Ischemic Stroke:
            alpha_amp_f3 = 25.0
            delta_amp_f3 = 7.0
            alpha_amp_f4 = 4.0
            delta_amp_f4 = 45.0

        elif self.patient_state == "STRESS":
            # High Cognitive Stress / Anxiety:
            # Symmetrical suppression of Alpha, pronounced elevation in Beta
            alpha_amp_f3, alpha_amp_f4 = 10.0, 10.0
            beta_amp_f3, beta_amp_f4 = 28.0, 28.0

        # Synthesize raw voltages for F3 (Left) and F4 (Right)
        f3_uV = (alpha_amp_f3 * np.sin(2 * np.pi * 10.2 * t) +
                 delta_amp_f3 * np.sin(2 * np.pi * 2.1 * t) +
                 beta_amp_f3  * np.sin(2 * np.pi * 21.5 * t) +
                 self._generate_pink_noise(n_samples))

        f4_uV = (alpha_amp_f4 * np.sin(2 * np.pi * 10.1 * t) +
                 delta_amp_f4 * np.sin(2 * np.pi * 2.0 * t) +
                 beta_amp_f4  * np.sin(2 * np.pi * 21.8 * t) +
                 self._generate_pink_noise(n_samples))

        # 2. PHYSIOLOGICAL ARTIFACT INJECTION=
        if self.inject_blink:
            # Low-frequency, large amplitude positive deflection (+120 uV) across both electrodes
            blink_wave = 120.0 * np.sin(np.linspace(0, np.pi, n_samples))
            f3_uV += blink_wave
            f4_uV += blink_wave
            self.inject_blink = False

        if self.inject_clench:
            # High-frequency, high-amplitude EMG muscle noise (+/- 70 uV)
            clench_noise = np.random.normal(0, 70.0, n_samples)
            f3_uV += clench_noise
            f4_uV += clench_noise
            self.inject_clench = False

        # 3. AD8232 ANALOG FRONT-END (AFE) FILTERING EMULATION
        # -------------------------------------------------------------
        # Replicates the analog hardware circuit filtering (0.5 Hz to 40 Hz bandpass)
        f3_conditioned = lfilter(self.filter_b, self.filter_a, f3_uV)
        f4_conditioned = lfilter(self.filter_b, self.filter_a, f4_uV)

        # 4. ATmega328P 10-BIT ADC QUANTIZATION
        # --------------------------------------------------------------
        # Biosignals are analog AC voltages centered around a 1.65V DC bias point (V_ref / 2).
        # Scaled so +/- 150 uV comfortably fits within the 10-bit integer range (0 to 1023).
        f3_centered = (f3_conditioned / 200.0) + 1.65
        f4_centered = (f4_conditioned / 200.0) + 1.65

        f3_adc = np.clip(np.round(f3_centered * (self.adc_resolution / self.v_ref)), 0, 1023).astype(int)
        f4_adc = np.clip(np.round(f4_centered * (self.adc_resolution / self.v_ref)), 0, 1023).astype(int)

        # Assemble the virtual hardware telemetry packet
        packet = {
            "timestamp": int(time.time() * 1000),
            "state": self.patient_state,
            "sampling_rate_hz": self.fs,
            "sample_count": n_samples,
            # Instantaneous representative voltages in microvolts
            "f3_uV_instant": round(float(f3_conditioned[-1]), 2),
            "f4_uV_instant": round(float(f4_conditioned[-1]), 2),
            # Full 10-bit integer raw arrays ready for DSP (sliding-window FFT)
            "f3_adc_raw": f3_adc.tolist(),
            "f4_adc_raw": f4_adc.tolist(),
            "hardware_battery_pct": 94,
            "electrode_impedance_kohm": {"F3": 4.8, "F4": 5.1, "Ear_Ground": 2.3}
        }
        return packet


# STANDALONE DEMONSTRATION & TEST HARNESS
if __name__ == "__main__":
    print("=" * 70)
    print("   COSMOS VIRTUAL HARDWARE DIGITAL TWIN (FRONTAL EEG HEADBAND)")
    print("=" * 70)
    
    headband = CosmosVirtualHardware(sampling_rate=256)
    print("[INIT] Dual AD8232 Front-End and ATmega328P ADC emulation running...\n")

    try:
        # Phase 1: Stream NORMAL baseline for 4 seconds
        print(">>> Phase 1: Simulating NORMAL Healthy Baseline (Bilateral balance)...")
        for i in range(12):
            pkt = headband.read_packet(duration_sec=0.25)
            print(f"[{pkt['timestamp']}] State: {pkt['state']:<12} | "
                  f"F3: {pkt['f3_uV_instant']:>6.2f} uV (ADC: {pkt['f3_adc_raw'][-1]:>4}) | "
                  f"F4: {pkt['f4_uV_instant']:>6.2f} uV (ADC: {pkt['f4_adc_raw'][-1]:>4}) | "
                  f"Battery: {pkt['hardware_battery_pct']}%")
            time.sleep(0.25)

        # Phase 2: Inject an Eye Blink Artifact
        print("\n>>> Phase 2: Simulating Patient Eye Blink Artifact...")
        headband.trigger_eye_blink()
        for i in range(4):
            pkt = headband.read_packet(duration_sec=0.25)
            print(f"[{pkt['timestamp']}] State: {pkt['state']:<12} | "
                  f"F3: {pkt['f3_uV_instant']:>6.2f} uV (ADC: {pkt['f3_adc_raw'][-1]:>4}) | "
                  f"F4: {pkt['f4_uV_instant']:>6.2f} uV (ADC: {pkt['f4_adc_raw'][-1]:>4})")
            time.sleep(0.25)

        # Phase 3: Trigger ACUTE LEFT MCA ISCHEMIC STROKE
        print("\n>>> Phase 3: Triggering ACUTE LEFT MCA ISCHEMIC STROKE (F3 drops, Delta spikes)...")
        headband.set_state("STROKE_LEFT")
        for i in range(16):
            pkt = headband.read_packet(duration_sec=0.25)
            print(f"[{pkt['timestamp']}] State: {pkt['state']:<12} | "
                  f"F3: {pkt['f3_uV_instant']:>6.2f} uV (ADC: {pkt['f3_adc_raw'][-1]:>4}) | "
                  f"F4: {pkt['f4_uV_instant']:>6.2f} uV (ADC: {pkt['f4_adc_raw'][-1]:>4})")
            time.sleep(0.25)

        print("\n[SUCCESS] Virtual Hardware Test Run Complete.")

    except KeyboardInterrupt:
        print("\n[STOP] Virtual Hardware halted by user.")
