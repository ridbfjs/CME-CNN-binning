"""Model B stage 1: existing Model A images + event-level physical features.

Keep dataset.py/data_process.py/model_loader.py/train.py unchanged.
Run alongside dataset.py, data_process.py, and physical_148_model_b.csv.
"""
import csv
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from dataset import CMEDataset

FEATURES = (
    'Linear_Speed_kms', 'Second_order_final_kms', 'Angular_Width_deg',
    'Central_PA', 'Acceleration_ms2', 'OMNI_Bx_nT_6h',
    'OMNI_Bz_nT_6h', 'OMNI_Temperature_K_6h',
    'OMNI_Speed_kms_6h', 'OMNI_Flow_Long_deg_6h',
    'OMNI_Flow_Pressure_nPa_6h',
)


def load_physics(path):
    """Load pre-exported 148 event CSV; enforce unique IDs and finite 11-vectors."""
    result = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            eid = int(row['event_id'])
            if eid in result:
                raise ValueError(f'Duplicate physics event_id {eid}')
            vec = np.asarray([float(row[k]) for k in FEATURES], dtype=np.float64)
            if vec.shape != (11,) or not np.isfinite(vec).all():
                raise ValueError(f'Invalid physics features for event {eid}')
            result[eid] = {
                'features': vec,
                'CME_time_UT': datetime.fromisoformat(row['CME_time_UT']),
                'arrival_time_UT': datetime.fromisoformat(row['arrival_time_UT']),
                'CPA_original': row.get('CPA_original', ''),
            }
    if not result:
        raise ValueError('Empty physics CSV')
    return result


class PhysicsScaler:
    """One scaler per train fold; unique training EVENT IDs, not all events."""
    def __init__(self, physics_table, train_event_ids):
        ids = sorted({int(e) for e in train_event_ids})
        if not ids:
            raise ValueError('No training event IDs')
        missing = set(ids).difference(physics_table)
        if missing:
            raise ValueError(f'Missing train event IDs: {sorted(missing)}')
        matrix = np.stack([physics_table[e]['features'] for e in ids])
        self.mean = matrix.mean(axis=0)
        self.scale = matrix.std(axis=0)
        self.scale[self.scale == 0] = 1.0
        self.training_event_ids = tuple(ids)

    def transform(self, raw):
        values = (np.asarray(raw, dtype=np.float64) - self.mean) / self.scale
        if not np.isfinite(values).all():
            raise ValueError('Invalid standardized features')
        return torch.tensor(values, dtype=torch.float32)


class CMEDatasetB(CMEDataset):
    """Return image[1,256,256], physics[11], label scalar (hours)."""
    def __init__(self, labels, physics_csv, indices=None, scaler=None):
        super().__init__(labels, indices)
        self.physics_table = load_physics(physics_csv)
        self.scaler = scaler
        for row in self.rows:
            eid = int(row['event_id'])
            if eid not in self.physics_table:
                raise ValueError(f'No physics row for event {eid}')
            physical = self.physics_table[eid]
            for k in ('CME_time_UT','arrival_time_UT'):
                from_manifest = datetime.fromisoformat(row[k])
                if abs((from_manifest - physical[k]).total_seconds()) > 1.01:
                    raise ValueError(f'Event {eid}: mismatch in {k}')

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        eid = int(self.rows[index]['event_id'])
        raw = self.physics_table[eid]['features']
        physics = (self.scaler.transform(raw) if self.scaler is not None
                   else torch.tensor(raw, dtype=torch.float32))
        return image, physics, label


def scaler_for_train_indices(labels, physics_csv, train_indices):
    """Fit unique-event train-only statistics on a split's image indices."""
    from data_process import read_csv
    rows = read_csv(labels)
    table = load_physics(physics_csv)
    event_ids = [int(rows[int(i)]['event_id']) for i in train_indices]
    return PhysicsScaler(table, event_ids)
