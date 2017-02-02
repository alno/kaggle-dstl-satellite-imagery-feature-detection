from .archs import unet

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
}
