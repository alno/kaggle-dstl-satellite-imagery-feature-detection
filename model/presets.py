from .archs import unet, unet2, unet3, unet_mi, unet_mi_2, unet_ma, unet3_mi, rnet1, rnet1_mi, rnet2, rnet2_mi, unet2_mi, unet_vgg16

presets = {
    'u1i': {
        'arch': unet,
        'n_epoch': 100,
        'mask_patch_size': 64,
        'batch_mode': 'random',
        'inputs': {
            'in': {'band': 'I'}
        }
    },

    'u1m': {
        'arch': unet,
        'n_epoch': 100,
        'mask_patch_size': 80,
        'mask_downscale': 4,
        'batch_mode': 'random',
        'batch_size': 32,
        'epoch_batches': 400,
        'inputs': {
            'in': {'band': 'M'}
        }
    },

    'u1ma': {
        'arch': unet_ma,
        'n_epoch': 100,
        'mask_patch_size': 80,
        'mask_downscale': 4,
        'batch_mode': 'random',
        'batch_size': 32,
        'epoch_batches': 400,
        'inputs': {
            'in_M': {'band': 'M'},
            'in_A': {'band': 'A'}
        }
    },

    'r1m': {
        'arch': rnet1,
        'n_epoch': 100,
        'mask_patch_size': 128,
        'mask_downscale': 4,
        'batch_mode': 'random',
        'batch_size': 32,
        'inputs': {
            'in': {'band': 'M'}
        }
    },

    'r1m_tmp': {
        'arch': rnet1,
        'n_epoch': 100,
        'mask_patch_size': 128,
        'mask_downscale': 4,
        'batch_mode': 'random',
        'batch_size': 32,
        'inputs': {
            'in': {'band': 'M'}
        }
    },

    'r2m': {
        'arch': rnet2,
        'n_epoch': 100,
        'mask_patch_size': 128,
        'mask_downscale': 4,
        'batch_mode': 'grid',
        'batch_size': 32,
        'inputs': {
            'in_I': {'band': 'I', 'downscale': 4},
            'in_IF': {'band': 'IF', 'downscale': 4},
            'in_M': {'band': 'M'}
        }
    },

    'r2m_tmp': {
        'arch': rnet2,
        'n_epoch': 100,
        'mask_patch_size': 128,
        'mask_downscale': 4,
        'batch_mode': 'random',
        'batch_size': 32,
        'inputs': {
            'in_I': {'band': 'I', 'downscale': 4},
            'in_IF': {'band': 'IF', 'downscale': 4},
            'in_M': {'band': 'M'}
        }
    },

    'r1mi': {
        'arch': rnet1_mi,
        'n_epoch': 100,
        'mask_patch_size': 128,
        'batch_mode': 'random',
        'batch_size': 32,
        'inputs': {
            'in_I': {'band': 'I'},
            'in_M': {'band': 'M'}
        }
    },

    'u2m': {
        'arch': unet2,
        'n_epoch': 100,
        'mask_patch_size': 128,
        'mask_downscale': 4,
        'batch_mode': 'random',
        'batch_size': 32,
        'inputs': {
            'in_I': {'band': 'I', 'downscale': 4},
            'in_IF': {'band': 'IF', 'downscale': 4},
            'in_M': {'band': 'M'}
        },
        'augment': {
            'channel_shift_range': 0.01,
            'channel_scale_range': 0.01
        }
    },

    'u2m_tmp': {
        'arch': unet2,
        'n_epoch': 100,
        'mask_patch_size': 128,
        'mask_downscale': 4,
        'batch_mode': 'random',
        'batch_size': 32,
        'inputs': {
            'in_I': {'band': 'I', 'downscale': 4},
            'in_IF': {'band': 'IF', 'downscale': 4},
            'in_M': {'band': 'M'}
        },
        'augment': {
            'channel_shift_range': 0.01,
            'channel_scale_range': 0.01
        }
    },

    'uvi': {
        'arch': unet_vgg16,
        'n_epoch': 100,
        'mask_patch_size': 128,
        'batch_mode': 'random',
        'batch_size': 32,
        'inputs': {
            'in_I': {'band': 'I'},
        },
        'augment': {
            'channel_shift_range': 0.01,
            'channel_scale_range': 0.01
        }
    },

    'uvi_structs': {
        'arch': unet_vgg16,
        'n_epoch': 150,
        'mask_patch_size': 128,
        'batch_mode': 'random',
        'batch_size': 32,
        'inputs': {
            'in_I': {'band': 'I'},
        },
        'augment': {
            'channel_shift_range': 0.01,
            'channel_scale_range': 0.01
        },
        'classes': [0, 1]
    },

    'u2mi_structs': {
        'arch': unet2_mi,
        'n_epoch': 100,
        'mask_patch_size': 128,
        'batch_mode': 'random',
        'batch_size': 32,
        'epoch_batches': 100,
        'inputs': {
            'in_I': {'band': 'I'},
            'in_IF': {'band': 'IF'},
            'in_M': {'band': 'M'}
        },
        'augment': {
            'channel_shift_range': 0.01,
            'channel_scale_range': 0.01
        },
        'classes': [0, 1, 2, 3]
    },

    'u3m': {
        'arch': unet3,
        'n_epoch': 100,
        'mask_patch_size': 112,
        'mask_downscale': 4,
        'batch_mode': 'grid',
        'batch_size': 32,
        'inputs': {
            'in': {'band': 'M'}
        }
    },

    'u3mi': {
        'arch': unet3_mi,
        'n_epoch': 100,
        'mask_patch_size': 112,
        'batch_mode': 'random',
        'batch_size': 32,
        'epoch_batches': 200,
        'inputs': {
            'in_I': {'band': 'I'},
            'in_M': {'band': 'M'}
        }
    },

    'u3mi_structs': {
        'arch': unet3_mi,
        'n_epoch': 100,
        'mask_patch_size': 112,
        'batch_mode': 'random',
        'batch_size': 32,
        'epoch_batches': 200,
        'inputs': {
            'in_I': {'band': 'I'},
            'in_M': {'band': 'M'}
        },
        'classes': [0, 1, 2, 3]
    },

    'u3m_areas': {
        'arch': unet3,
        'n_epoch': 100,
        'mask_patch_size': 112,
        'mask_downscale': 4,
        'batch_mode': 'random',
        'batch_size': 32,
        'epoch_batches': 200,
        'inputs': {
            'in': {'band': 'M'}
        },
        'classes': [4, 5, 6, 7]
    },

    'r1m_areas': {
        'arch': rnet1,
        'n_epoch': 100,
        'mask_patch_size': 128,
        'mask_downscale': 4,
        'batch_mode': 'random',
        'batch_size': 32,
        'epoch_batches': 200,
        'inputs': {
            'in': {'band': 'M'}
        },
        'classes': [4, 5, 6, 7]
    },

    'u3mi_cars': {
        'arch': unet3_mi,
        'n_epoch': 100,
        'mask_patch_size': 112,
        'batch_mode': 'random',
        'batch_size': 32,
        'epoch_batches': 200,
        'inputs': {
            'in_I': {'band': 'I'},
            'in_M': {'band': 'M'}
        },
        'classes': [8, 9]
    },

    'r1mi_cars': {
        'arch': rnet1_mi,
        'n_epoch': 100,
        'mask_patch_size': 64,
        'batch_mode': 'random',
        'batch_size': 32,
        'epoch_batches': 600,
        'inputs': {
            'in_I': {'band': 'I'},
            'in_M': {'band': 'M'}
        },
        'classes': [8, 9]
    },

    'r2mi_water': {
        'arch': rnet2_mi,
        'n_epoch': 100,
        'mask_patch_size': 128,
        'batch_mode': 'random',
        'batch_size': 32,
        'epoch_batches': 300,
        'inputs': {
            'in_I': {'band': 'I'},
            'in_IF': {'band': 'IF'},
            'in_M': {'band': 'M'}
        },
        'classes': [6, 7]
    },

    'umi': {
        'arch': unet_mi_2,
        'n_epoch': 40,
        'mask_patch_size': 160,
        'batch_mode': 'random',
        'batch_size': 32,
        'epoch_batches': 100,
        'inputs': {
            'in_I': {'band': 'I'},
            'in_M': {'band': 'M'}
        }
    },

    'umi_struct': {
        'arch': unet_mi,
        'n_epoch': 50,
        'mask_patch_size': 80,
        'mask_downscale': 1,
        'batch_mode': 'random',
        'batch_size': 48,
        'inputs': {
            'in_I': {'band': 'I'},
            'in_M': {'band': 'M'}
        },
        'classes': [0, 1, 2, 3]
    },

    'um_areas': {
        'arch': unet,
        'n_epoch': 50,
        'mask_patch_size': 64,
        'mask_downscale': 4,
        'batch_mode': 'random',
        'inputs': {
            'in': {'band': 'M'}
        },
        'classes': [4, 5, 6, 7]
    },

    'umi_cars': {
        'arch': unet_mi,
        'n_epoch': 50,
        'mask_patch_size': 48,
        'mask_downscale': 1,
        'batch_mode': 'random',
        'batch_size': 48,
        'inputs': {
            'in_I': {'band': 'I'},
            'in_M': {'band': 'M'}
        },
        'classes': [8, 9]
    },
}
