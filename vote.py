import pandas as pd

import shapely.wkt

from util.data import sample_submission

from shapely.ops import unary_union
from shapely.geometry import MultiPolygon

subm_names = [
    'kostia-merge',
    'subm-r2m_tmp-20170212-1322-fix',
    'subm-d8m-20170228-0537'
]

classes = [3]

min_total_area = 1e-8

print "Loading subms..."

subms = [pd.read_csv('subm/%s.csv.gz' % s) for s in subm_names]

subm = sample_submission.copy()

for image_id in sorted(subm['ImageId'].unique()):
    print "%s..." % image_id

    subm.loc[subm['ImageId'] == image_id, 'MultipolygonWKT'] = 'GEOMETRYCOLLECTION EMPTY'

    for cls in classes:
        polys = [shapely.wkt.loads(s.loc[(s['ImageId'] == image_id) & (s['ClassType'] == cls), 'MultipolygonWKT'].iloc[0]) for s in subms]

        poly_parts = []

        for i in xrange(len(polys)):
            for j in xrange(i+1, len(polys)):
                poly_parts.append(polys[i].intersection(polys[j]))

        res = unary_union(poly_parts)

        if res.area < min_total_area:
            continue

        if res.type == 'Polygon':
            res = MultiPolygon([res])

        subm.loc[(subm['ImageId'] == image_id) & (subm['ClassType'] == cls), 'MultipolygonWKT'] = shapely.wkt.dumps(res, rounding_precision=8)

print "Saving..."
subm.to_csv('subm/vote-%s-%s.csv.gz' % ('+'.join(subm_names), '+'.join(map(str, classes))), compression='gzip', index=False)

print "Done."
