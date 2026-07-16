# preprocessing/motion_fft/

Computes Fast Fourier Transform (FFT) features from accelerometer signals, producing the frequency-domain representation used as input to the LocalGlobalLSTM.

## Files

| File | Purpose |
|---|---|
| `motion_fft_service.py` | Segments a motion channel into 30-second epochs and computes FFT for each |
| `motion_fft_collection.py` | Holds the resulting FFT array (epochs × FFT bins) for one channel |
| `motion_fft_feature_service.py` | Saves and loads FFT feature `.out` files; manages file paths |

## What this produces

For each accelerometer channel (x, y, z, and vector magnitude), the FFT is computed over each 30-second epoch at the raw sampling rate. Frequency bins 1–30 Hz are extracted, giving a 30-element feature vector per epoch per channel.

The four resulting feature arrays for a single subject have shape `(N_epochs, 30)` and are stored as:

```
features/<subject_id>_motion_xfft1_30.out
features/<subject_id>_motion_yfft1_30.out
features/<subject_id>_motion_zfft1_30.out
features/<subject_id>_motion_vmfft1_30.out
```

These four arrays are stacked into the final model input of shape `(N_epochs, 30, 4)` per subject.

## Why FFT?

Sleep-wake classification from actigraphy is well suited to frequency-domain features because
the dominant difference between sleep and wake is in the *rhythm* of movement (periodic breathing
and body micro-movements during sleep vs. voluntary gross motor activity during wake), not just
its absolute magnitude. The LocalGlobalLSTM processes these spectral features at both a local
(per-epoch) and global (across-the-night) level.
