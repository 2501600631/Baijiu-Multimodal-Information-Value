from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
TARGETS = dict(acetaldehyde='Acetaldehyde', methanol='Methanol', n_propanol='n-Propanol', ethyl_acetate='Ethyl acetate', isobutanol='Isobutanol', acetal='Acetal', n_butanol='n-Butanol', isoamyl_alcohol='Isoamyl alcohol', ethyl_butyrate='Ethyl butyrate', ethyl_lactate='Ethyl lactate', ethyl_hexanoate='Ethyl hexanoate')

@dataclass
class Dataset:
    ids: np.ndarray
    X: dict
    axes: dict
    Y: pd.DataFrame
    metadata: pd.DataFrame
    audit: dict

    @property
    def target_names(self):
        return list(self.Y.columns)

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()

def load_dataset(data_dir, strict=True):
    folder = Path(data_dir)
    chemistry = pd.read_csv(folder / 'chemistry.csv', dtype={'sample_id': str}, float_precision='round_trip')
    if list(chemistry.columns) != ['sample_id'] + list(TARGETS):
        raise ValueError('Chemistry columns must retain the declared eleven-target order')
    ids = chemistry.sample_id.to_numpy(str)
    if len(set(ids)) != len(ids) or any((not v for v in ids)):
        raise ValueError('Missing or duplicate bottle identities')
    Y = chemistry[list(TARGETS)].copy()
    if not np.isfinite(Y.to_numpy(float)).all() or np.any(Y.to_numpy(float) < 0):
        raise ValueError('Invalid reference concentrations')
    X = {}
    axes = {}
    sources = {}
    for modality in ('uv', 'raman', 'nir'):
        path = folder / ('spectra_' + modality + '.npz')
        with np.load(path, allow_pickle=False) as z:
            source_ids = z['sample_id'].astype(str)
            spectrum = z['spectra'].copy()
            axis = z['axis'].copy()
        if len(set(source_ids)) != len(source_ids) or set(source_ids) != set(ids):
            raise ValueError('Spectral bottle identities do not match reference concentrations')
        if spectrum.shape != (len(source_ids), len(axis)) or not np.isfinite(spectrum).all():
            raise ValueError('Invalid spectrum dimensions or nonfinite data')
        if not np.isfinite(axis).all() or not np.all(np.diff(axis) > 0):
            raise ValueError('Spectral coordinates must be finite and strictly increasing')
        lookup = {v: i for i, v in enumerate(source_ids)}
        X[modality] = spectrum[[lookup[v] for v in ids]]
        axes[modality] = axis
        sources[modality] = dict(file=path.name, sha256=sha256(path), n_features=len(axis), axis_min=float(axis.min()), axis_max=float(axis.max()), axis_step_min=float(np.diff(axis).min()), axis_step_max=float(np.diff(axis).max()), axis_units='Legacy exported coordinates; confirm physical units against instrument records')
    Y.index = pd.Index(ids, name='sample_id')
    summary = [dict(target=t, source_unit='g/L', n=len(ids), min=float(Y[t].min()), max=float(Y[t].max()), mean=float(Y[t].mean()), sd_ddof1=float(Y[t].std(ddof=1))) for t in TARGETS]
    audit = dict(observational_unit='Author-confirmed independent bottle', n_matched_bottles=len(ids), n_targets=11, strict_alignment=True, sources=sources, target_summary=summary, concentration_unit='g/L', provisional_endpoint='The ethyl_hexanoate identifier retains the legacy endpoint mapping pending author confirmation', mutations='Lossless storage and English schema conversion only; no values imputed, corrected or deleted')
    return Dataset(ids, X, axes, Y, pd.DataFrame(index=Y.index), audit)
