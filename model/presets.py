from .archs import unet, unet2, unet3, unet_mi, unet_mi_2, unet_ma, unet3_mi, rnet1, rnet1_mi, rnet2, rnet2_mi, unet2_mi, unet_vgg16, unet_water, rnet3, unet4_mi, dnet1_mi, dnet2, dnet3, dnet4, rnet4, dnet5, rnet5, dnet6, dnet7, dnet8

from keras.optimizers import Adam

import numpy as np

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

    'r2m_areas': {
        'arch': rnet2,
        'n_epoch': 70,
        'mask_patch_size': 128,
        'mask_downscale': 4,
        'batch_mode': 'random',
        'batch_size': 48,
        'inputs': {
            'in_I': {'band': 'I', 'downscale': 4},
            'in_IF': {'band': 'IF', 'downscale': 4},
            'in_M': {'band': 'M'},
            'in_MI': {'band': 'MI'},
        },
        'classes': [4, 5, 6, 7]
    },

    'r2m_2': {
        'arch': rnet2,
        'n_epoch': 70,
        'mask_patch_size': 128,
        'mask_downscale': 4,
        'batch_mode': 'random',
        'batch_size': 48,
        'inputs': {
            'in_I': {'band': 'I', 'downscale': 4},
            'in_IF': {'band': 'IF', 'downscale': 4},
            'in_M': {'band': 'M'},
            'in_MI': {'band': 'MI'},
        },
        'classes': [0, 1, 2, 3, 4, 5]
    },

    'r3m_1': {
        'arch': rnet3,
        'n_epoch': 70,
        'mask_patch_size': 128,
        'mask_downscale': 4,
        'batch_mode': 'random',
        'batch_size': 48,
        'inputs': {
            'in_I': {'band': 'I', 'downscale': 4},
            'in_IF': {'band': 'IF', 'downscale': 4},
            'in_M': {'band': 'M'},
            'in_MI': {'band': 'MI'},
        },
        'classes': [0, 1, 2, 3, 4, 5]
    },

    'r3m_2': {
        'arch': rnet3,
        'n_epoch': 70,
        'mask_patch_size': 128,
        'mask_downscale': 4,
        'batch_mode': 'random',
        'batch_size': 48,
        'inputs': {
            'in_I': {'band': 'I', 'downscale': 4},
            'in_IF': {'band': 'IF', 'downscale': 4},
            'in_M': {'band': 'M'},
            'in_MI': {'band': 'MI'},
        },
        'augment': {
            'mirror': False,
            'transpose': False
        },
        'classes': [0, 1, 2, 3, 4, 5]
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

    'u_water': {
        'arch': unet_water,
        'n_epoch': 100,
        'mask_patch_size': 128,
        'mask_downscale': 4,
        'batch_mode': 'random',
        'batch_size': 32,
        'inputs': {
            'in_I': {'band': 'I', 'downscale': 4},
            'in_IF': {'band': 'IF', 'downscale': 4},
            'in_M': {'band': 'M'},
            'in_MI': {'band': 'MI'},
        },
        'augment': {
            'channel_shift_range': 1e-4,
            'channel_scale_range': 1e-4
        },
        'classes': [6, 7]
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

    'u4mi_str': {
        'arch': unet4_mi,
        'n_epoch': 100,
        'mask_patch_size': 112,
        'batch_mode': 'random',
        'batch_size': 32,
        'epoch_batches': 200,
        'inputs': {
            'in_I': {'band': 'I'},
            'in_IF': {'band': 'IF'},
            'in_M': {'band': 'M'},
            'in_MI': {'band': 'MI'},
        },
        'augment': {
            'mirror': False,
            'transpose': False
        },
        'classes': [0, 1, 4]
    },

    'u4mi_str_2': {
        'arch': unet4_mi,
        'n_epoch': 100,
        'mask_patch_size': 160,
        'batch_mode': 'random',
        'batch_size': 32,
        'epoch_batches': 100,
        'inputs': {
            'in_I': {'band': 'I'},
            'in_IF': {'band': 'IF'},
            'in_M': {'band': 'M'},
            'in_MI': {'band': 'MI'},
        },
        'augment': {
            'mirror': False,
            'transpose': False
        },
        'classes': [0, 1, 4],
#        'optimizer': Adam(1e-2, decay=4e-4)
    },

    'd1mi': {
        'arch': dnet1_mi,
        'mask_patch_size': 160,
        'inputs': {
            'in_I': {'band': 'I'},
            'in_IF': {'band': 'IF'},
            'in_M': {'band': 'M'},
            'in_MI': {'band': 'MI'},
        },
        'train': [{
            'n_epoch': 100,
            'epoch_batches': 100,
            'batch_size': 32,
            'augment': {
                'mirror': False,
                'transpose': False
            },
            'optimizer': Adam(1e-3, decay=4e-4),
        }, {
            'n_epoch': 50,
            'epoch_batches': 100,
            'batch_size': 32,
            'optimizer': Adam(1e-4, decay=4e-4),
        }]
    },

    'd1mi_str_2': {
        'arch': dnet1_mi,
        'mask_patch_size': 160,
        'inputs': {
            'in_I': {'band': 'I'},
            'in_IF': {'band': 'IF'},
            'in_M': {'band': 'M'},
            'in_MI': {'band': 'MI'},
        },
        'classes': [0, 1, 4],
        'train': [{
            'n_epoch': 40,
            'epoch_batches': 100,
            'batch_size': 32,
            'augment': {
                'mirror': False,
                'transpose': False
            },
            'optimizer': Adam(1e-4, decay=4e-4),
        }]
    },

    'd2_1': {
        'arch': dnet2,
        'mask_patch_size': 128,
        'mask_downscale': 4,
        'inputs': {
            'd0_I': {'band': 'I', 'downscale': 4},
            'd0_IF': {'band': 'IF', 'downscale': 4},
            'd0_M': {'band': 'M'},
            'd0_MI': {'band': 'MI'},
        },
        'classes': [0, 1, 2, 3, 4, 5, 6, 7],
        'train': [{
            'n_epoch': 300,
            'epoch_batches': 100,
            'batch_size': 32,
            'augment': {
                'mirror': False,
                'transpose': False
            },
            'optimizer': Adam(1e-4, decay=2e-4),
        }]
    },

    'd3_1': {
        'arch': dnet3,
        'mask_patch_size': 128,
        'mask_downscale': 4,
        'inputs': {
            'd0_I': {'band': 'I', 'downscale': 4},
            'd0_IF': {'band': 'IF', 'downscale': 4},
            'd0_M': {'band': 'M'},
            'd0_MI': {'band': 'MI'},
        },
        'classes': [2, 3, 4, 5, 6, 7],
        'train': [{
            'n_epoch': 10,
            'epoch_batches': 100,
            'batch_size': 16,
            'augment': {
                'mirror': True,
                'transpose': True,
            },
            'optimizer': Adam(1e-1, decay=2e-4),
            'loss_jac_weight': 0.05,
        }, {
            'n_epoch': 100,
            'epoch_batches': 100,
            'batch_size': 32,
            'augment': {
                'mirror': True,
                'transpose': True,
            },
            'optimizer': Adam(1e-2, decay=2e-4),
            'loss_jac_weight': 0.15,
        }, {
            'n_epoch': 300,
            'epoch_batches': 100,
            'batch_size': 32,
            'augment': {
                'mirror': True,
                'transpose': True,
            },
            'optimizer': Adam(1e-4, decay=2e-4),
            'loss_jac_weight': 0.3,
        }]
    },

    'd3_2': {
        'arch': dnet3,
        'mask_patch_size': 128,
        'mask_downscale': 4,
        'inputs': {
            'd0_I': {'band': 'I', 'downscale': 4},
            'd0_IF': {'band': 'IF', 'downscale': 4},
            'd0_M': {'band': 'MN'},
            'd0_MI': {'band': 'MI'},
        },
        'classes': [2, 3, 4, 5, 6, 7],
        'train': [{
            'val_only': True,
            'n_epoch': 20,
            'epoch_batches': 100,
            'batch_size': 16,
            'augment': {
                'mirror': True,
                'transpose': True,
            },
            'optimizer': Adam(1e-1, decay=2e-4),
            'loss_jac_weight': 0.05,
        }, {
            'n_epoch': 100,
            'epoch_batches': 100,
            'batch_size': 32,
            'augment': {
                'mirror': True,
                'transpose': True,
            },
            'optimizer': Adam(1e-2, decay=2e-4),
            'loss_jac_weight': 0.15,
        }, {
            'n_epoch': 100,
            'epoch_batches': 100,
            'batch_size': 32,
            'augment': {
                'mirror': True,
                'transpose': True,
            },
            'optimizer': Adam(1e-4, decay=2e-4),
            'loss_jac_weight': 0.3,
        }]
    },

    'd3_str': {
        'arch': dnet3,
        'mask_patch_size': 128,
        'inputs': {
            'd0_I': {'band': 'I'},
            'd0_IF': {'band': 'IF'},
            'd2_M': {'band': 'MN'},
            'd2_MI': {'band': 'MI'},
        },
        'classes': [0, 1, 2, 3, 4, 8, 9],
        'train': [{
            'val_only': True,
            'n_epoch': 20,
            'epoch_batches': 100,
            'batch_size': 16,
            'augment': {
                'mirror': False,
                'transpose': False,
            },
            'optimizer': Adam(1e-1, decay=2e-4),
            'loss_jac_weight': 0.05,
        }, {
            'n_epoch': 100,
            'epoch_batches': 100,
            'batch_size': 32,
            'augment': {
                'mirror': False,
                'transpose': False,
            },
            'optimizer': Adam(1e-2, decay=2e-4),
            'loss_jac_weight': 0.15,
        }, {
            'n_epoch': 100,
            'epoch_batches': 100,
            'batch_size': 32,
            'augment': {
                'mirror': False,
                'transpose': False,
            },
            'optimizer': Adam(1e-4, decay=2e-4),
            'loss_jac_weight': 0.3,
        }]
    },

    'd4_areas': {
        'arch': dnet4,
        'arch_options': {'softmax': False},
        'mask_patch_size': 128,
        'mask_downscale': 4,
        'inputs': {
            'd0_I': {'band': 'I', 'downscale': 4},
            'd0_IF': {'band': 'IF', 'downscale': 4},
            'd0_M': {'band': 'MN'},
            'd0_MI': {'band': 'MI'},
        },
        'classes': [5, 6, 7],
        'train': [{
            'val_only': True,
            'n_epoch': 20,
            'epoch_batches': 100,
            'batch_size': 16,
            'augment': {
                'mirror': True,
                'transpose': True,
            },
            'optimizer': Adam(1e-1, decay=2e-4),
            'loss_jac_weight': 0.05,
        }, {
            'n_epoch': 100,
            'epoch_batches': 100,
            'batch_size': 32,
            'augment': {
                'mirror': True,
                'transpose': True,
            },
            'optimizer': Adam(1e-2, decay=2e-4),
            'loss_jac_weight': 0.15,
        }, {
            'n_epoch': 100,
            'epoch_batches': 100,
            'batch_size': 32,
            'augment': {
                'mirror': True,
                'transpose': True,
            },
            'optimizer': Adam(1e-4, decay=2e-4),
            'loss_jac_weight': 0.3,
        }]
    },

    'r4_areas': {
        'arch': rnet4,
        'mask_patch_size': 128,
        'mask_downscale': 4,
        'inputs': {
            'd0_I': {'band': 'I', 'downscale': 4},
            'd0_IF': {'band': 'IF', 'downscale': 4},
            'd0_M': {'band': 'M'},
            'd0_MI': {'band': 'MI'},
        },
        'classes': [2, 3, 4, 5, 6, 7],
        'train': [{
            'n_epoch': 100,
            'epoch_batches': 100,
            'batch_size': 32,
            'augment': {
                'mirror': True,
                'transpose': True,
            },
            'optimizer': Adam(1e-2, decay=2e-4),
            'loss_jac_weight': 0.15,
        }]
    },

    'd5_areas': {
        'arch': dnet5,
        'mask_patch_size': 80,
        'mask_downscale': 8,
        'inputs': {
            'd0_I': {'band': 'I', 'downscale': 4},
            'd0_IF': {'band': 'IF', 'downscale': 4},
            'd0_M': {'band': 'M'},
            'd0_MI': {'band': 'MI'},
        },
        'classes': [2, 3, 4, 5, 6, 7],
        'train': [{
            'n_epoch': 100,
            'epoch_batches': 100,
            'batch_size': 32,
            'augment': {
                'mirror': True,
                'transpose': True,
            },
            'optimizer': Adam(1e-2, decay=2e-4),
            'loss_jac_weight': 0.15,
        }]
    },

    'r5_cars': {
        'arch': rnet5,
        'mask_patch_size': 128,
        'inputs': {
            'd0_I': {'band': 'I'},
            'd0_IF': {'band': 'IF'},
            'd2_M': {'band': 'M'},
            'd2_MI': {'band': 'MI'},
        },
        'classes': [0, 2, 8, 9],
        'train': [{
            'n_epoch': 50,
            'epoch_batches': 100,
            'batch_size': 32,
            'augment': {
                'mirror': False,
                'transpose': False,
            },
            'optimizer': Adam(1e-2, decay=2e-4),
            'loss_jac_weight': 0.15,
            'batch_class_threshold': np.array([10, 10, 1, 1]),
            'class_weights': np.array([0.2, 0.2, 1, 1]),
        }, {
            'n_epoch': 100,
            'epoch_batches': 100,
            'batch_size': 32,
            'augment': {
                'mirror': False,
                'transpose': False,
            },
            'optimizer': Adam(1e-3, decay=2e-4),
            'loss_jac_weight': 0.15,
            'class_weights': np.array([0.1, 0.1, 1, 1]),
        }, {
            'n_epoch': 200,
            'epoch_batches': 100,
            'batch_size': 32,
            'augment': {
                'mirror': False,
                'transpose': False,
            },
            'optimizer': Adam(1e-3, decay=2e-4),
            'loss_jac_weight': 0.5,
            'class_weights': np.array([0.1, 0.1, 1, 1]),
        }]
    },

    'd6_cars': {
        'arch': dnet6,
        'arch_options': {'block_size': 24},
        'mask_patch_size': 128,
        'inputs': {
            'd0_I': {'band': 'I'},
            'd0_IF': {'band': 'IF'},
            'd2_M': {'band': 'M'},
            'd2_MI': {'band': 'MI'},
        },
        'classes': [0, 2, 8, 9],
        'train': [{
            'n_epoch': 50,
            'epoch_batches': 100,
            'batch_size': 32,
            'augment': {
                'mirror': False,
                'transpose': False,
                'channel_shift_range': 1e-3,
                'channel_scale_range': 1e-2,
            },
            'optimizer': Adam(1e-2, decay=2e-4),
            'loss_jac_weight': 0.15,
            'batch_class_threshold': np.array([10, 10, 1, 1]),
            'class_weights': np.array([0.2, 0.2, 1, 1]),
            'ema': False,
        }, {
            'n_epoch': 100,
            'epoch_batches': 100,
            'batch_size': 32,
            'augment': {
                'mirror': False,
                'transpose': False,
                'channel_shift_range': 1e-3,
                'channel_scale_range': 1e-2,
            },
            'optimizer': Adam(5e-3, decay=2e-4),
            'loss_jac_weight': 0.15,
            'class_weights': np.array([0.1, 0.1, 1, 1]),
            'ema': False,
        }]
    },

    'r5_cars_2': {
        'arch': rnet5,
        'mask_patch_size': 128,
        'inputs': {
            'd0_I': {'band': 'I'},
            'd0_IF': {'band': 'IF'},
            'd2_M': {'band': 'M'},
            'd2_MI': {'band': 'MI'},
        },
        'normalization': 'std',
        'classes': [0, 2, 3, 8, 9],
        'train': [{
            'n_epoch': 50,
            'epoch_batches': 100,
            'batch_size': 32,
            'augment': {
                'mirror': True,
                'transpose': True,
                #'channel_shift_range': 1e-3,
                #'channel_scale_range': 1e-2,
                'rotation': 5
            },
            'optimizer': Adam(1e-2, decay=2e-4),
            'loss_jac_weight': 0.15,
            'batch_class_threshold': np.array([1e7, 1e7, 1e7, 1, 1]),
            'class_weights': np.array([0.1, 0.1, 0.1, 1, 1]),
        }, {
            'n_epoch': 100,
            'epoch_batches': 100,
            'batch_size': 32,
            'augment': {
                'mirror': True,
                'transpose': True,
                #'channel_shift_range': 1e-3,
                #'channel_scale_range': 1e-2,
                'rotation': 5
            },
            'optimizer': Adam(5e-3, decay=2e-4),
            'loss_jac_weight': 0.15,
            'class_weights': np.array([0.05, 0.05, 0.05, 1, 1]),
        }]
    },

    'd7_cars_2': {
        'arch': dnet7,
        'mask_patch_size': 128,
        'inputs': {
            'd0_I': {'band': 'I'},
            'd0_IF': {'band': 'IF'},
            'd2_M': {'band': 'M'},
            'd2_MI': {'band': 'MI'},
        },
        'normalization': 'std',
        'classes': [0, 2, 3, 8, 9],
        'train': [{
            'n_epoch': 50,
            'epoch_batches': 100,
            'batch_size': 32,
            'augment': {
                'mirror': True,
                'transpose': True,
                'channel_shift_range': 1e-3,
                'channel_scale_range': 1e-2,
                'rotation': 5
            },
            'optimizer': Adam(1e-2, decay=2e-4),
            'loss_jac_weight': 0.15,
            'batch_class_threshold': np.array([1e7, 1e7, 1e7, 1, 1]),
            'class_weights': np.array([0.1, 0.1, 0.1, 1, 1]),
        }, {
            'n_epoch': 100,
            'epoch_batches': 100,
            'batch_size': 32,
            'augment': {
                'mirror': True,
                'transpose': True,
                'channel_shift_range': 1e-3,
                'channel_scale_range': 1e-2,
                'rotation': 5
            },
            'optimizer': Adam(5e-3, decay=2e-4),
            'loss_jac_weight': 0.15,
            'class_weights': np.array([0.05, 0.05, 0.05, 1, 1]),
        }]
    },

    'r5_trucks': {
        'arch': rnet5,
        'mask_patch_size': 128,
        'mask_downscale': 4,
        'inputs': {
            'd0_I': {'band': 'I', 'downscale': 4},
            'd0_IF': {'band': 'IF', 'downscale': 4},
            'd0_M': {'band': 'M'},
        },
        'normalization': 'std',
        'classes': [0, 2, 3, 8],
        'train': [{
            'n_epoch': 50,
            'epoch_batches': 100,
            'batch_size': 32,
            'augment': {
                'mirror': False,
                'transpose': False,
                #'channel_shift_range': 1e-3,
                #'channel_scale_range': 1e-2,
                'rotation': 5
            },
            'optimizer': Adam(1e-2, decay=2e-4),
            'loss_jac_weight': 0.15,
            'batch_class_threshold': np.array([1e7, 1e7, 1e7, 1]),
            'class_weights': np.array([0.1, 0.1, 0.1, 3.0]),
        }, {
            'n_epoch': 100,
            'epoch_batches': 100,
            'batch_size': 32,
            'augment': {
                'mirror': False,
                'transpose': False,
                #'channel_shift_range': 1e-3,
                #'channel_scale_range': 1e-2,
                'rotation': 5
            },
            'optimizer': Adam(5e-3, decay=2e-4),
            'loss_jac_weight': 0.15,
            'class_weights': np.array([0.05, 0.05, 0.05, 3.0]),
        }]
    },

    'd7_cars_3': {
        'arch': dnet7,
        'mask_patch_size': 96,
        'inputs': {
            'd0_I': {'band': 'I'},
            'd0_IF': {'band': 'IF'},
            'd2_M': {'band': 'M'},
        },
        'normalization': 'std',
        'classes': [8, 9],
        'train': [{
            'n_epoch': 150,
            'epoch_batches': 100,
            'batch_size': 32,
            'augment': {
                'mirror': True,
                'transpose': True,
                'channel_shift_range': 0,
                'channel_scale_range': 0,
                'rotation': 7
            },
            'optimizer': Adam(1e-2, decay=2e-4),
            'loss_jac_weight': 0.9,
            'batch_class_threshold': np.array([1, 1]),
            'batch_noclass_accept_proba': 0.00,
            'batch_noclass_accept_proba_growth': 1e-4,
            'class_weights': np.array([1, 1]),
        }]
    },

    'd8m': {
        'arch': dnet8,
        'mask_patch_size': 128,
        'mask_downscale': 4,
        'inputs': {
            'd0_IF': {'band': 'IF', 'downscale': 4},
            'd0_M': {'band': 'MN'},
        },
        'normalization': 'std',
        'train': [{
            'n_epoch': 160,
            'epoch_batches': 100,
            'batch_size': 32,
            'augment': {
                'mirror': True,
                'transpose': True,
                'channel_shift_range': 1e-4,
                'channel_scale_range': 0,
                'rotation': 5
            },
            'optimizer': Adam(1e-2, decay=2e-4),
            'loss_jac_weight': 0.5,
        }]
    },

    'd8m_roads': {
        'arch': dnet8,
        'init': 'd8m',
        'mask_patch_size': 128,
        'mask_downscale': 4,
        'inputs': {
            'd0_IF': {'band': 'IF', 'downscale': 4},
            'd0_M': {'band': 'MN'},
        },
        'normalization': 'std',
        'train': [{
            'n_epoch': 100,
            'epoch_batches': 100,
            'batch_size': 32,
            'augment': {
                'mirror': True,
                'transpose': True,
                'channel_shift_range': 1e-4,
                'channel_scale_range': 0,
                'rotation': 5
            },
            'optimizer': Adam(1e-3, decay=2e-4),
            'loss_jac_weight': 0.5,
            'class_weights': np.array([0.1, 0.1, 2.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        }]
    },

    'd8m_crops': {
        'arch': dnet8,
        'init': 'd8m',
        'mask_patch_size': 128,
        'mask_downscale': 4,
        'inputs': {
            'd0_IF': {'band': 'IF', 'downscale': 4},
            'd0_M': {'band': 'MN'},
        },
        'normalization': 'std',
        'train': [{
            'n_epoch': 100,
            'epoch_batches': 100,
            'batch_size': 32,
            'augment': {
                'mirror': True,
                'transpose': True,
                'channel_shift_range': 1e-4,
                'channel_scale_range': 0,
                'rotation': 5
            },
            'optimizer': Adam(1e-3, decay=2e-4),
            'loss_jac_weight': 0.5,
            'class_weights': np.array([0.1, 0.1, 0.1, 0.1, 0.1, 2.0, 0.1, 0.1, 0.1, 0.1])
        }]
    },

    'd8mi': {
        'arch': dnet8,
        'mask_patch_size': 128,
        'inputs': {
            'd0_I': {'band': 'I'},
            'd0_IF': {'band': 'IF'},
            'd2_M': {'band': 'MN'},
        },
        'normalization': 'std',
        'train': [{
            'n_epoch': 150,
            'epoch_batches': 100,
            'batch_size': 32,
            'augment': {
                'mirror': True,
                'transpose': True,
                'channel_shift_range': 1e-4,
                'channel_scale_range': 0,
                'rotation': 5
            },
            'optimizer': Adam(1e-2, decay=2e-4),
            'loss_jac_weight': 0.5,
        }]
    },

    'd8mi_build_structs': {
        'arch': dnet8,
        'init': 'd8mi',
        'mask_patch_size': 128,
        'inputs': {
            'd0_I': {'band': 'I'},
            'd0_IF': {'band': 'IF'},
            'd2_M': {'band': 'MN'},
        },
        'normalization': 'std',
        'train': [{
            'n_epoch': 150,
            'epoch_batches': 100,
            'batch_size': 32,
            'augment': {
                'mirror': True,
                'transpose': True,
                'channel_shift_range': 1e-4,
                'channel_scale_range': 0,
                'rotation': 5
            },
            'optimizer': Adam(1e-3, decay=2e-4),
            'loss_jac_weight': 0.5,
            'class_weights': np.array([1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        }]
    },

    'r2m_3': {
        'arch': rnet2,
        'mask_patch_size': 160,
        'mask_downscale': 4,
        'inputs': {
            'in_I': {'band': 'I', 'downscale': 4},
            'in_IF': {'band': 'IF', 'downscale': 4},
            'in_M': {'band': 'M'},
            'in_MI': {'band': 'MI'}
        },
        'train': [{
            'n_epoch': 100,
            'epoch_batches': 100,
            'batch_size': 32,
        }, {
            'n_epoch': 50,
            'epoch_batches': 100,
            'batch_size': 32,
            'optimizer': Adam(3e-4, decay=4e-4),
        }]
    },
}
