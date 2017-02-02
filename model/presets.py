from .archs import unet, unet_mi

presets = {
    'u1i': {
        'arch': unet,
        'n_epoch': 100,
        'mask_patch_size': 64,
        'mask_downscale': 1,
        'batch_mode': 'random',
        'inputs': {
            'in': {'band': 'I'}
        }
    },

    'u1m': {
        'arch': unet,
        'n_epoch': 100,
        'mask_patch_size': 64,
        'mask_downscale': 4,
        'batch_mode': 'random',
        'inputs': {
            'in': {'band': 'M'}
        }
    },

    'umi': {
        'arch': unet_mi,
        'n_epoch': 100,
        'mask_patch_size': 80,
        'mask_downscale': 1,
        'batch_mode': 'random',
        'batch_size': 32,
        'inputs': {
            'in_I': {'band': 'I'},
            'in_M': {'band': 'M'}
        }
    }
}
