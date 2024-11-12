import os
import argparse
from utils.config import parse
from train import Trainer

def defineyaml(args):
    f_name = args.name
    yamlcontents = f"""
f_name: '{f_name}'
model_name: deeplabv3plus
model_path: hfrs/checkpoints/hfrs_model_best.pth.tar
train: true
model_specs:
    encoder_name: efficientnet-b0
    in_channels: 3
    classes: 19
    upsampling: 4
batch_size: 32
data_specs:
    width: 512
    height: 512
    dtype:
    image_type: 32bit
    rescale: false
    rescale_minima: auto
    rescale_maxima: auto
    label_type: mask
    is_categorical: true
    mask_channels: 1
    val_holdout_frac:
    data_workers: 4
training_data_csv: {args.traincsv}
validation_data_csv: {args.validcsv}
training_augmentation:
    augmentations:
        HorizontalFlip:
            p: 0.5
        RandomScale:
            scale_limit: [-0.3,0.3]
        PadIfNeeded:
            min_height: 512
            min_width: 512
        CenterCrop:
            height: 512
            width: 512
        #ChannelShuffle:
        #RandomRotate90:
        #    p: 0.5
        # RandomCrop:
        #     height: 512
        #     width: 512
        #     p: 1.0
        # Normalize:
        #     mean:
        #         - 0.5
        #     std:
        #         - 0.125
        #     max_pixel_value: 255.0
        #     p: 1.0
    p: 1.0
    shuffle: true
validation_augmentation:
    augmentations:
        # Resize:
        #     height: 512
        #     width: 512
        #     p: 1.0
        # CenterCrop:
        #     height: 512
        #     width: 512
        #     p: 1.0
        # Normalize:
        #     mean:
        #         - 0.5
        #     std:
        #         - 0.125
        #     max_pixel_value: 255.0
        #     p: 1.0
    p: 1.0
training:
    epochs: 200
    lr: 5e-2
    loss:
        diceloss:
            mode: multiclass
            from_logits: True
        crossentropyloss:
    loss_weights:
        diceloss: 1.0
        crossentropyloss: 1.0
    """
    print('saving file name is ', f_name)
    checkpoint_dir = os.path.join('./', f_name, 'checkpoints')
    logs_dir = os.path.join('./', f_name, 'logs')
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.isdir(logs_dir):
        os.makedirs(logs_dir)

    with open(os.path.join('./', f_name, f'{f_name}.yaml'), 'w') as f:
        f.write(yamlcontents)

    return f_name, checkpoint_dir, logs_dir

def main(args):
    f_name, ckpt_dir, logs_dir = defineyaml(args)
    config = parse(os.path.join('./', f_name, f'{f_name}.yaml'))
    config['ckpt_dir'] = ckpt_dir
    config['logs_dir'] = logs_dir
    trainer = Trainer(config)
    trainer.run()
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deepv3 Face Parsing')
    parser.add_argument('--traincsv', default='./train.csv',
                        help='Where to save reference CSV of training data')
    parser.add_argument('--validcsv', default='./valid.csv',
                        help='Where to save reference CSV of validation data')
    parser.add_argument('--name', default='face_init',
                        help='Name of the output folder')
    args = parser.parse_args()

    main(args)