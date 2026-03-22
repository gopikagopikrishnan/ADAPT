def tofc_mapping(
    rf_data:       np.ndarray,   # (n_z, N_ELEMENTS) normalised RF
    fs:            float,
    probe_geometry: np.ndarray,  # (N_ELEMENTS,) lateral positions [m]
    c:             float,
) -> np.ndarray:
    """Time-of-Flight Correction via GPU-vectorised bilinear interpolation.

    Returns (n_z, N_ELEMENTS, N_ELEMENTS) on CPU, float64.
    """
    num_samples, num_channels = rf_data.shape
    x_np = np.asarray(probe_geometry, dtype=np.float64).ravel()

    rf = cp.asarray(rf_data, dtype=cp.float64, order="C")
    x  = cp.asarray(x_np,   dtype=cp.float64)
    t  = cp.arange(num_samples, dtype=cp.float64) / float(fs)
    z  = 0.5 * c * t

    xg, zg  = cp.meshgrid(x, z, indexing="xy")
    x_flat  = xg.ravel()
    z_flat  = zg.ravel()

    dx       = x_flat[:, None] - x[None, :]
    rx       = cp.sqrt(dx * dx + z_flat[:, None] ** 2)
    td       = (z_flat[:, None] + rx) / c

    ir       = cp.searchsorted(t, td, side="right")
    il       = cp.clip(ir - 1, 0, num_samples - 2)
    ir       = il + 1

    t_l      = t[il];  t_r = t[ir]
    w        = (td - t_l) / cp.maximum(t_r - t_l, 1e-6)

    ch_idx   = cp.arange(num_channels, dtype=cp.int64)[None, :]
    y        = (1.0 - w) * rf[il, ch_idx] + w * rf[ir, ch_idx]
    y[(td < t[0]) | (td > t[-1])] = 0.0

    return cp.asnumpy(y.reshape(num_samples, num_channels, num_channels))
