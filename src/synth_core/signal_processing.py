import numpy as np
import numpy.typing as npt
from typing import Any, Tuple


def add_gaussian_noise(data: npt.NDArray[np.float64], snr: float) -> npt.NDArray[np.float64]:
    """
    Add Gaussian noise to the data based on desired SNR.
    """
    signal_power = np.mean(data**2)
    noise_power = signal_power / snr
    noise = np.random.normal(0, np.sqrt(noise_power), data.shape)
    return data + noise


def compute_fsc(
    data1: npt.NDArray[Any],
    data2: npt.NDArray[Any],
    voxel_size: tuple[float, float, float],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute the Fourier Shell Correlation (FSC) between two 3D maps.
    Returns frequencies and correlation values.
    """
    assert data1.shape == data2.shape

    # Fourier transforms in float64 for precision
    # Use rfftn instead of fftn for real inputs.
    # This not only improves performance but also circumvents a memory-corruption
    # bug in NumPy 2.2.6 with Python 3.14 during complex multiplication.
    f1 = np.fft.rfftn(data1)
    f2 = np.fft.rfftn(data2)

    # Cross-spectral density and power spectra are computed using pure float arithmetic.
    # We avoid `f1 * np.conj(f2)` and `np.real(f1 * np.conj(f1))` because the complex
    # multiplication triggers a severe memory mutation/aliasing bug in Numpy 2.2.6 prerelease.
    cross = f1.real * f2.real + f1.imag * f2.imag
    p1 = f1.real**2 + f1.imag**2
    p2 = f2.real**2 + f2.imag**2

    # Calculate radial bins
    nz, ny, nx = data1.shape
    kz = np.fft.fftfreq(nz, d=voxel_size[0])
    ky = np.fft.fftfreq(ny, d=voxel_size[1])
    kx = np.fft.rfftfreq(nx, d=voxel_size[2])

    # Create 3D grid of frequencies
    kz_grid, ky_grid, kx_grid = np.meshgrid(kz, ky, kx, indexing="ij")

    # Calculate magnitude of frequency vector for each voxel
    k = np.sqrt(kz_grid**2 + ky_grid**2 + kx_grid**2)

    # Flatten everything
    k = k.ravel()
    cross = cross.ravel()
    p1 = p1.ravel()
    p2 = p2.ravel()

    # Sort by frequency
    idx = np.argsort(k)
    k_sorted = k[idx]
    cross_sorted = cross[idx]
    p1_sorted = p1[idx]
    p2_sorted = p2[idx]

    # Binning — start at a small epsilon so the DC voxel (k=0) is excluded.
    # Including DC would produce FSC=1.0 in whichever bin it lands in, which is
    # misleading and whose position is sensitive to floating-point precision in
    # np.linspace across NumPy builds.
    n_bins = min(nx, ny, nz) // 2
    k_max = float(k_sorted.max())
    k_eps = k_max / (10 * n_bins)  # one-tenth of a bin width
    bins = np.linspace(k_eps, k_max, n_bins + 1)

    fsc = []
    freqs = []

    for i in range(n_bins):
        mask = (k_sorted >= bins[i]) & (k_sorted < bins[i + 1])
        if np.any(mask):
            c_bin = cross_sorted[mask]
            p1_bin = p1_sorted[mask]
            p2_bin = p2_sorted[mask]

            # Sum of cross power and individual powers (using float64 accumulation)
            sum_cross = np.sum(c_bin)
            sum_p1 = np.sum(p1_bin)
            sum_p2 = np.sum(p2_bin)

            # FSC is real part of cross correlation / sqrt(power1 * power2)
            num = np.real(sum_cross)
            den = np.sqrt(sum_p1 * sum_p2)

            if den > 0:
                # Clamp to [-1, 1] to avoid numerical issues
                val = num / den
                fsc.append(np.clip(val, -1.0, 1.0))
                freqs.append((bins[i] + bins[i + 1]) / 2.0)

    freqs_arr = np.array(freqs, dtype=np.float64)
    fsc_arr = np.array(fsc, dtype=np.float64)

    # Guarantee output is sorted by ascending frequency (defensive against any
    # future NumPy sort-stability changes).
    order = np.argsort(freqs_arr, kind="stable")
    return freqs_arr[order], fsc_arr[order]
