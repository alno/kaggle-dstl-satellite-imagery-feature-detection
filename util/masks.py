import numpy as np
import cv2

import shapely.affinity

from shapely.geometry import MultiPolygon, Polygon
from collections import defaultdict


def convert_geo_coords_to_raster(coords, raster_size, xymax):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    Xmax, Ymax = xymax
    H, W = raster_size
    W1 = 1.0 * W * W / (W + 1)
    H1 = 1.0 * H * H / (H + 1)
    xf = W1 / Xmax
    yf = H1 / Ymax
    coords[:, 1] *= yf
    coords[:, 0] *= xf
    coords_int = np.round(coords).astype(np.int32)
    return coords_int


def convert_geo_poly_to_raster_contours(poly_list, raster_size, xymax):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    perim_list = []
    interior_list = []

    if poly_list is None:
        return None

    for k in range(len(poly_list)):
        poly = poly_list[k]
        perim = np.array(list(poly.exterior.coords))
        perim_c = convert_geo_coords_to_raster(perim, raster_size, xymax)
        perim_list.append(perim_c)

        for pi in poly.interiors:
            interior = np.array(list(pi.coords))
            interior_c = convert_geo_coords_to_raster(interior, raster_size, xymax)
            interior_list.append(interior_c)

    return perim_list, interior_list


def plot_contours(raster_size, contours, class_value=1):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    img_mask = np.zeros(raster_size, np.uint8)

    if contours is None:
        return img_mask

    perim_list, interior_list = contours
    cv2.fillPoly(img_mask, perim_list, class_value)
    cv2.fillPoly(img_mask, interior_list, 0)

    return img_mask


def poly_to_mask(poly, raster_size, xymax):
    contours = convert_geo_poly_to_raster_contours(poly, raster_size, xymax)
    return plot_contours(raster_size, contours, 1)


def get_scalers(im_size, x_max, y_min):
    # __author__ = Konstantin Lopuhin
    # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly
    h, w = im_size  # they are flipped so that mask_for_polygons works correctly
    h, w = float(h), float(w)
    w_ = 1.0 * w * (w / (w + 1))
    h_ = 1.0 * h * (h / (h + 1))
    return w_ / x_max, h_ / y_min


def convert_poly_to_geo_coords(poly, raster_size, xymax):
    x_max, y_min = xymax
    x_scaler, y_scaler = get_scalers(raster_size, x_max, y_min)

    return shapely.affinity.scale(poly, xfact=1.0 / x_scaler, yfact=1.0 / y_scaler, origin=(0, 0, 0))


def mask_to_poly(mask, xymax, epsilon=2, min_area=1., threshold=0.5):
    # __author__ = Konstantin Lopuhin
    # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly

    # first, find contours with cv2: it's much faster than shapely
    contours, hierarchy = cv2.findContours(((mask >= threshold) * 255).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)

    # create approximate contours to have reasonable submission size
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True) for cnt in contours]

    if not contours:
        return MultiPolygon()

    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1

    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])

    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)

    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(all_polygons).buffer(0)

    # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
    # need to keep it a Multi throughout
    if all_polygons.type == 'Polygon':
        all_polygons = MultiPolygon([all_polygons])

    return convert_poly_to_geo_coords(all_polygons, mask.shape, xymax)
