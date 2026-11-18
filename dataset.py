"""Each image has its own label in hours."""
import math
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from data_process import read_csv
class CMEDataset(Dataset):
    def __init__(self,labels,indices=None):
        self.base_dir=Path(labels).resolve().parent
        rows=read_csv(labels); self.rows=rows if indices is None else [rows[int(i)] for i in indices]
        if not self.rows: raise ValueError('Empty dataset')
        for row in self.rows:
            hours=float(row['transit_time_hours'])
            expected=(datetime.fromisoformat(row['arrival_time_UT'])-datetime.fromisoformat(row['image_time_UT'])).total_seconds()/3600
            if not math.isfinite(hours) or hours<0 or abs(hours-expected)>1e-6: raise ValueError(f'Invalid label: {row}')
            image_path=Path(row['processed_image_path'])
            if not image_path.is_absolute(): image_path=self.base_dir/image_path
            row['processed_image_path']=str(image_path)
            if not image_path.is_file(): raise FileNotFoundError(image_path)
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        with Image.open(row['processed_image_path']) as image:
            if image.mode!='L' or image.size!=(256,256): raise ValueError('Expected preprocessed 256x256 L image')
            array=np.array(image,dtype=np.float32)/255.0
        return torch.from_numpy(array).unsqueeze(0),torch.tensor(float(row['transit_time_hours']),dtype=torch.float32)
