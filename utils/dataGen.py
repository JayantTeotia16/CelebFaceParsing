import numpy as np
from torch.utils.data import Dataset, DataLoader

from utils.core import _check_df_load
from utils.transform import _check_augs
from utils.io import imread, _check_channel_order

def make_data_generator(config, df, stage = 'train'):
    df = _check_df_load(df)
    if stage == 'train':
        augs = config['training_augmentation']
        shuffle = config['training_augmentation']['shuffle']
        batch_size=config['batch_size']
    elif stage == 'val':
        augs = config['validation_augmentation']
        shuffle = False
        batch_size=1

    try:
        num_classes = config['model_specs']['classes']
    except KeyError:
        num_classes = 1

    dataset = TorchDataset(
            df,
            augs=augs,
            batch_size=batch_size,
            label_type=config['data_specs']['label_type'],
            is_categorical=config['data_specs']['is_categorical'],
            num_classes=num_classes,
            dtype=config['data_specs']['dtype'])

    data_workers = config['data_specs'].get('data_workers')
    if data_workers == 1 or data_workers is None:
        data_workers = 0
    
    data_gen = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=data_workers
    )

    return data_gen

class TorchDataset(Dataset):
    def __init__(self, df, augs, batch_size, label_type='mask',
                 is_categorical=False, num_classes=1, dtype=None):
        super().__init__()

        self.df = df
        self.batch_size = batch_size
        self.n_batches = int(np.floor(len(self.df)/self.batch_size))
        self.aug = _check_augs(augs)
        self.is_categorical = is_categorical
        self.num_classes = num_classes
        if dtype is None:
            self.dtype = np.float32
        elif isinstance(dtype, str):
            try:
                self.dtype = getattr(np, dtype)
            except AttributeError:
                raise ValueError(
                    'The data type {} is not supported'.format(dtype))
        elif issubclass(dtype, np.number) or isinstance(dtype, np.dtype):
            self.dtype = dtype

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image = imread(self.df['image'].iloc[idx])
        mask = imread(self.df['label'].iloc[idx])
        if not self.is_categorical:
            mask[mask != 0] = 1
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]

        sample = {'image': image, 'mask':mask}#, 'path': self.df['image'].iloc[idx]}
        if self.aug:
            sample = self.aug(**sample)
        
        sample['image'] = _check_channel_order(sample['image'], 'torch').astype(self.dtype)
        return sample