import pandas as pd

import sys

import shapely.wkt

from util.data import sample_submission, grid_sizes
from util.masks import mask_to_poly, poly_to_mask

from shapely.ops import unary_union
from shapely.geometry import MultiPolygon


presets = {
    1: {
        'subms': [
            'unet-rot-15-custom-dice-10-aligned-12-cls-0-1-model-26-cls-0',
            'unet3l-noaug-custom-dice-10-aligned-12-cls-0-1-model-12-lr0.000002-eps1-cls0',
            'unet-fullaug-square-dice-10-bn-eps-2-cls-0',
        ]
    },

    2: {
        'subms': [
            'unet-rot-15-custom-dice-10-aligned-12-cls-0-1-model-26-cls-1',
            'subm-d1mi_str_2-20170224-1815-fix-filtered-2',
            'objects_flip_12000_2',
        ]
    },

    3: {
        'subms': [
            'unet-fullaug-all-dice-10-cls-2-3-cls-2',
            'unet-fullaug-dice-10-cls-2-3.csv-model-64-cls-2',
            'unet-fullaug-square-dice-10-bn-eps-2-cls-2',
        ],
    },

    4: {
        'subms': [
            'objects_14400_4',
            'subm-multi-20170207-0943-rnd',
            'subm-r2m_3-20170303-2036',
        ],
        #'pre_buffer_size': 1e-5,
        #'post_buffer_size': -1e-5,
        'pixelize': True,
    },

    5: {
        'subms': [
            'subm-r2m_tmp-20170212-1322-fix',
            'subm-u4mi_str-20170220-1125-fix',
            'subm-d1mi_str_2-20170224-1815-fix',
        ],
        'pixelize': True,
    },

    9: {
        'subms': [
            'subm-r2m_tmp-20170212-1322-fix-filtered-9',
            'unet-fullaug-square-dice-10-bn-eps-2-cls-8-9',
            'subm-d1mi-20170304-0604'
        ]
    },
}

cls = int(sys.argv[1])
preset = presets[cls]

subm_names = preset['subms']
classes = [cls]

pre_buffer_size = preset.get('pre_buffer_size')
post_buffer_size = preset.get('post_buffer_size')

min_total_area = preset.get('min_total_area', 0)

pixelize = preset.get('pixelize', False)

print "Loading subms..."

subms = [pd.read_csv('subm/%s.csv.gz' % s) for s in subm_names]

subm = sample_submission.copy()

for image_id in sorted(subm['ImageId'].unique()):
    print "%s..." % image_id

    xymax = (grid_sizes.loc[image_id, 'xmax'], grid_sizes.loc[image_id, 'ymin'])

    subm.loc[subm['ImageId'] == image_id, 'MultipolygonWKT'] = 'MULTIPOLYGON EMPTY'

    for cls in classes:
        polys = [shapely.wkt.loads(s.loc[(s['ImageId'] == image_id) & (s['ClassType'] == cls), 'MultipolygonWKT'].iloc[0]) for s in subms]

        if pre_buffer_size is not None:
            polys = [p.buffer(pre_buffer_size) for p in polys]

        if pixelize:
            mask = sum(poly_to_mask(p, (18000, 18000), xymax) for p in polys) > len(polys) * 0.5

            if 'mask_postprocess' in preset:
                mask = preset['mask_postprocess'](mask)

            res = mask_to_poly(mask, xymax, min_area=1.0, threshold=1.0)
        else:
            try:
                poly_parts = []

                for i in xrange(len(polys)):
                    for j in xrange(i+1, len(polys)):
                        poly_parts.append(polys[i].intersection(polys[j]))

                res = unary_union(poly_parts)
            except:
                print "Error, using first poly"
                res = polys[0]

        if post_buffer_size is not None:
            res = res.buffer(post_buffer_size)

        res = res.buffer(0)

        if res.area < min_total_area:
            continue

        if res.type == 'Polygon':
            res = MultiPolygon([res])

        if not res.is_valid:
            raise ValueError("Invalid geometry")

        subm.loc[(subm['ImageId'] == image_id) & (subm['ClassType'] == cls), 'MultipolygonWKT'] = shapely.wkt.dumps(res, rounding_precision=9)

print "Saving..."
subm_name = 'vote-%s-%s' % ('+'.join(map(str, classes)), '+'.join(subm_names))
subm.to_csv('subm/%s.csv.gz' % subm_name, compression='gzip', index=False)

print "Done, %s" % subm_name
