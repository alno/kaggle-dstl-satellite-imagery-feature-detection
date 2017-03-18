import pandas as pd

import sys

import shapely.wkt

from util.data import sample_submission, grid_sizes
from util.masks import mask_to_poly, poly_to_mask

from shapely.ops import unary_union
from shapely.geometry import MultiPolygon

presets = {
    4: {
        'subms': [
            'objects_14400_4',
            'subm-multi-20170207-0943-rnd',
        ],
        'pixelize': True
    },

    9: {
        'subms': [
            'subm-r2m_tmp-20170212-1322-fix-filtered-9',
            'unet-fullaug-square-dice-10-bn-eps-2-cls-8-9',
        ]
    },
}

cls = int(sys.argv[1])
preset = presets[cls]

subm_names = preset['subms']
classes = [cls]
buffer_size = preset.get('buffer_size', 0)
min_total_area = preset.get('min_total_area', 0)

print "Loading subms..."

subms = [pd.read_csv('subm/%s.csv.gz' % s) for s in subm_names]

subm = sample_submission.copy()

for image_id in sorted(subm['ImageId'].unique()):
    print "%s..." % image_id

    xymax = (grid_sizes.loc[image_id, 'xmax'], grid_sizes.loc[image_id, 'ymin'])

    subm.loc[subm['ImageId'] == image_id, 'MultipolygonWKT'] = 'MULTIPOLYGON EMPTY'

    for cls in classes:
        polys = [shapely.wkt.loads(s.loc[(s['ImageId'] == image_id) & (s['ClassType'] == cls), 'MultipolygonWKT'].iloc[0]) for s in subms]

        if preset.get('pixelize', False):
            mask = sum(poly_to_mask(p, (9000, 9000), xymax) for p in polys) > 0.5

            if 'mask_postprocess' in preset:
                mask = preset['mask_postprocess'](mask)

            res = mask_to_poly(mask, xymax, min_area=1.0, threshold=1.0)
        else:
            try:
                res = unary_union(polys)
            except:
                print "Error, using first poly"
                res = polys[0]

        res = res.buffer(0)

        if res.area < min_total_area:
            continue

        if res.type == 'Polygon':
            res = MultiPolygon([res])

        if not res.is_valid:
            raise ValueError("Invalid geometry")

        subm.loc[(subm['ImageId'] == image_id) & (subm['ClassType'] == cls), 'MultipolygonWKT'] = shapely.wkt.dumps(res, rounding_precision=9)

print "Saving..."
subm_name = 'union-%s-%s' % ('+'.join(map(str, classes)), '+'.join(subm_names))
subm.to_csv('subm/%s.csv.gz' % subm_name, compression='gzip', index=False)

print "Done, %s" % subm_name
