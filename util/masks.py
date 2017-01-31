import numpy as np
import cv2


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
