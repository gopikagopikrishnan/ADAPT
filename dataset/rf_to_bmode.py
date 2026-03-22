import os
import numpy as np
import h5py
import psutil
import cupy as cp
from ultraspy.probes.factory import get_probe
from ultraspy.beamformers.das import DelayAndSum
from ultraspy.beamformers.fdmas import FilteredDelayMultiplyAndSum
from ultraspy.beamformers.capon import Capon
from ultraspy.scan import GridScan

# Configuration
# Path to the downloaded Zenodo dataset (zenodo.org/records/7813791)
input_dir  = "/content/wrist_data/wrist_data"   # change to your local path
output_dir = "/content/bmode_dataset_DAS_FDMAS_Capon"
os.makedirs(output_dir, exist_ok=True)

N_FILES       = 1000       # number of files to process
fs            = 31_250_000
fc            =  7_812_500
c             = 1540
threshold_gb  = 11          # RAM warning threshold in GB

def check_ram_usage():
    ram_used_gb = psutil.virtual_memory().used / (1024 ** 3)
    if ram_used_gb > threshold_gb:
        print(f"[WARNING] RAM usage: {ram_used_gb:.2f} GB  (>{threshold_gb} GB threshold)")


def pybeamforming(beamformer, data, probe, transmit_delays):
    """Run beamforming and return (b_mode, x, z)."""
    elements_indices  = np.arange(probe.nb_elements)
    nb_transmissions  = transmit_delays.shape[0]

    acquisition_info = {
        'sampling_freq':    fs,
        't0':               0,
        'prf':              1000,
        'signal_duration':  None,
        'delays':           transmit_delays,
        'sound_speed':      c,
        'sequence_elements': {
            'emitted':  np.tile(elements_indices, (nb_transmissions, 1)),
            'received': np.tile(elements_indices, (nb_transmissions, 1)),
        },
    }

    beamformer.automatic_setup(acquisition_info, probe)
    beamformer.update_setup('f_number', 1.75)
    beamformer.update_option('reduction',            'sum')
    beamformer.update_option('rx_apodization',       'boxcar')
    beamformer.update_option('rx_apodization_alpha', '0.5')
    beamformer.update_option('compound',             'True')

    x    = np.linspace(probe.geometry[0, 0], probe.geometry[0, -1], 128)
    z    = np.linspace(0, 0.06, data.shape[-1])
    scan = GridScan(x, z)

    d_data     = cp.asarray(data, dtype=cp.float32)
    d_output   = beamformer.beamform(d_data, scan)
    d_envelope = beamformer.compute_envelope(d_output, scan)
    beamformer.to_b_mode(d_envelope, scan)   # in-place log-compression
    b_mode = d_envelope.get()
    return b_mode, x, z


def safe_scale_and_cast(array, label):
    """Sanitise → int16 with a returned scale_factor for lossless inversion."""
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    scale = np.max(np.abs(array))

    if scale == 0 or not np.isfinite(scale):
        print(f"[WARNING] Zero/invalid scale for '{label}' — filling with zeros.")
        return np.zeros_like(array, dtype=np.int16), 1.0

    scale_factor  = scale / np.iinfo(np.int16).max
    scaled        = np.nan_to_num(array / scale_factor, nan=0.0, posinf=0.0, neginf=0.0)
    return scaled.astype(np.int16), scale_factor


# Main loop
for i in range(1, N_FILES + 1):
    file_path   = os.path.join(input_dir,  f"data_{i}.h5")
    output_path = os.path.join(output_dir, f"data_{i}.h5")

    try:
        check_ram_usage()

        # Load raw RF
        with h5py.File(file_path, 'r') as f:
            rawData        = np.array(f['data'], dtype=np.float32)   # [128, 2176]
            rawData        = np.moveaxis(rawData, [0, 1], [1, 0])    # [2176, 128]
            parameters     = dict(f['/data'].attrs.items())
            probe_name     = parameters['probe_name']
            transmit_delays = parameters['transmit_delays']
            if transmit_delays.ndim < 2:
                transmit_delays = np.expand_dims(transmit_delays, axis=0)

        probe = get_probe(probe_name)
        probe.set_central_freq(fc)

        # data shape expected by ultraspy: [n_transmissions, n_elements, n_samples]
        data = np.expand_dims(np.moveaxis(rawData, [1, 0], [0, 1]), axis=0)  # [1, 128, 2176]

        # Beamformign and bmode
        bmode_das,   x, z = pybeamforming(DelayAndSum(),                data, probe, transmit_delays)
        bmode_fdmas, x, z = pybeamforming(FilteredDelayMultiplyAndSum(), data, probe, transmit_delays)
        bmode_capon, x, z = pybeamforming(Capon(),                      data, probe, transmit_delays)

        # Scale and cast
        das_int16,   scale_das   = safe_scale_and_cast(np.real(bmode_das),   "DAS")
        fdmas_int16, scale_fdmas = safe_scale_and_cast(np.real(bmode_fdmas), "FDMAS")
        capon_int16, scale_capon = safe_scale_and_cast(np.real(bmode_capon), "Capon")

        # Write output to the output path
        with h5py.File(output_path, 'w') as hf:
            # Raw input
            hf.create_dataset('raw_data',       data=data.astype(np.float32), compression="gzip")

            # B-mode outputs
            hf.create_dataset('bmode_DAS',   data=das_int16,   compression="gzip")
            hf.create_dataset('bmode_FDMAS', data=fdmas_int16, compression="gzip")
            hf.create_dataset('bmode_Capon', data=capon_int16, compression="gzip")

            # Scale factors (multiply back to recover float range)
            hf.attrs['scale_DAS']   = scale_das
            hf.attrs['scale_FDMAS'] = scale_fdmas
            hf.attrs['scale_Capon'] = scale_capon

            # Metadata
            hf.attrs['probe_name']           = probe_name
            hf.create_dataset('probe_geometry', data=probe.geometry)
            hf.create_dataset('fs',             data=fs)
            hf.create_dataset('tx_delays',      data=transmit_delays)
            hf.create_dataset('scan_x',         data=x)
            hf.create_dataset('scan_z',         data=z)

        print(f"[INFO]  Saved: data_{i}.h5")

    except Exception as e:
        print(f"[ERROR] data_{i}.h5 — {e}")
