import numpy as np
import cv2
__all__ = ['normalize_shape','generate_output', 'restore_quandrangle']


def normalize_shape(image, polys, image_shape=(736,1280,3)):
    """
    이미지와 좌표들을 배율에 나누어 떨어지도록 정규화
    """
    h, w = image.shape[:2]
    new_h, new_w = image_shape[:2]

    ratio_h = new_h / h
    ratio_w = new_w / w

    polys = polys.astype(float)
    polys[:, :, 0] = polys[:, :, 0] * ratio_w  # x축에 배율 적용
    polys[:, :, 1] = polys[:, :, 1] * ratio_h  # y축에 배율 적용

    image = cv2.resize(image, (int(new_w), int(new_h)))
    return image, polys


def reorder_clockwise(polys):
    """
    4개 점들의 순서를 top-left -> top-right -> bottom-right -> bottom-left 즉,
    시계 방향으로 회전하도록 순서를 변경
    """
    if polys.ndim == 2:
        x_sorted = polys[np.argsort(polys[:, 0]), :]

        leftmost = x_sorted[:2, :]
        leftmost = leftmost[np.argsort(leftmost[:, 1]), :]
        (tl, bl) = leftmost

        rightmost = x_sorted[2:, :]
        rightmost = rightmost[np.argsort(rightmost[:, 1]), :]
        (tr, br) = rightmost

        return np.array([tl, tr, br, bl], dtype=np.float32)
    elif polys.ndim == 3:
        return np.stack([reorder_clockwise(poly) for poly in polys])
    else:
        raise ValueError("Wrong Dimension")


def get_reference_length(polys):
    """
    polys : (batch size, 4, 2)
    return :
    rs : (batch size, 4)
    """
    d_1 = np.sqrt(
        np.sum((polys - np.roll(polys, 1, axis=1))**2,
               axis=-1))
    d_2 = np.sqrt(
        np.sum((polys - np.roll(polys, -1, axis=1))**2,
               axis=-1))
    rs = np.min([d_1, d_2], axis=0)
    return rs


def shrink_quandrangle(poly, r, shrink_ratio=0.3):
    dists = np.linalg.norm(poly - np.roll(poly, -1, axis=0),
                           axis=1)

    def diff(theta): return np.array([np.cos(theta), np.sin(theta)])

    if dists[0] + dists[2] > dists[1] + dists[2]:
        theta = angle(poly[1], poly[0])
        poly[0] += shrink_ratio * r[0] * diff(theta)
        poly[1] -= shrink_ratio * r[1] * diff(theta)

        theta = angle(poly[2], poly[3])
        poly[3] += shrink_ratio * r[3] * diff(theta)
        poly[2] -= shrink_ratio * r[2] * diff(theta)

        theta = angle(poly[3], poly[0])
        poly[0] += shrink_ratio * r[0] * diff(theta)
        poly[3] -= shrink_ratio * r[3] * diff(theta)

        theta = angle(poly[2], poly[1])
        poly[1] += shrink_ratio * r[1] * diff(theta)
        poly[2] -= shrink_ratio * r[2] * diff(theta)
    else:
        theta = angle(poly[3], poly[0])
        poly[0] += shrink_ratio * r[0] * diff(theta)
        poly[3] -= shrink_ratio * r[3] * diff(theta)

        theta = angle(poly[2], poly[1])
        poly[1] += shrink_ratio * r[1] * diff(theta)
        poly[2] -= shrink_ratio * r[2] * diff(theta)

        theta = angle(poly[1], poly[0])
        poly[0] += shrink_ratio * r[0] * diff(theta)
        poly[1] -= shrink_ratio * r[1] * diff(theta)

        theta = angle(poly[2], poly[3])
        poly[3] += shrink_ratio * r[3] * diff(theta)
        poly[2] -= shrink_ratio * r[2] * diff(theta)

    return poly


def calculate_point_line_distance(pt1, pt2, pt0):
    """
    한 점과 2 점을 잇는 직선 간의 거리를 계산

    :param pt1, pt2: 직선을 구성하는 2개의 점
    :param pt0: (x0, y0) 목표 점
    :return:
        직선과 두 점의 수직 거리를 계산
    """
    x1, y1 = pt1
    x2, y2 = pt2
    x0, y0 = pt0
    return (np.abs((x2 - x1) * (y0 - y1) - (y2 - y1) * (x0 - x1))
            / np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))


def revise_points(points):
    """
    사다리꼴로 되어 있는 점들의 위치를 직사각형 형태로 바꾸어 줆

    :param points:
    :return:
    """
    mean_points = (points + np.roll(points, -1, axis=0)) / 2
    w0 = np.sqrt(np.sum((mean_points[0] - mean_points[2]) ** 2))
    w1 = np.sqrt(np.sum((mean_points[1] - mean_points[3]) ** 2))

    if w0 >= w1:
        h0 = calculate_point_line_distance(mean_points[0], mean_points[2],
                                           mean_points[1])
        h1 = calculate_point_line_distance(mean_points[0], mean_points[2],
                                           mean_points[3])
        vector = mean_points[0] - mean_points[2]
    else:
        h0 = calculate_point_line_distance(mean_points[1], mean_points[3],
                                           mean_points[0])
        h1 = calculate_point_line_distance(mean_points[1], mean_points[3],
                                           mean_points[2])
        vector = mean_points[1] - mean_points[3]

    theta = np.arctan2(vector[1], vector[0])
    if theta < 0:
        theta += np.pi
    theta = np.clip(theta, 0, np.pi)

    width = max(w0, w1)  # 항상 긴 쪽을 width
    height = max(h0, h1) * 2

    center = points.mean(axis=0)
    origins = np.array([[width / 2, height / 2],
                        [-width / 2, height / 2],
                        [-width / 2, -height / 2],
                        [width / 2, -height / 2]])

    transposed_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]])

    rotated_points = np.matmul(transposed_matrix, origins.T).T
    points = rotated_points + center
    return reorder_clockwise(points)


def generate_feature_map(image, polys, scale=4):
    height, width = image.shape[:2]
    h = height // scale
    w = width // scale

    score_map = np.zeros((h, w), dtype=np.uint8)
    geo_map = np.zeros((h, w, 5), dtype=np.float32)
    poly_mask = np.zeros((h, w), dtype=np.uint8)

    polys = polys.copy() / 4
    return score_map, geo_map, poly_mask, polys


def angle(pt1, pt2):
    return np.arctan2(pt1[1] - pt2[1], pt1[0] - pt2[0])


def generate_output(image, polys, fm_scale=4):
    # 특징 맵의 크기에 맞게 score_map, geo_map 만들기
    score_map, geo_map, poly_mask, polys = generate_feature_map(
        image, polys, fm_scale)

    # 점 위치 정렬하기
    polys = reorder_clockwise(polys)
    # Reference 길이 맞추기
    rs = get_reference_length(polys)

    for poly_idx, (poly, r) in enumerate(zip(polys, rs)):
        inner_poly = shrink_quandrangle(poly.copy(), r)

        cv2.fillPoly(score_map, [inner_poly.astype(np.int32)], 1)
        cv2.fillPoly(poly_mask, [inner_poly.astype(np.int32)], poly_idx + 1)

        coords_in_poly = np.argwhere(poly_mask == (poly_idx + 1))

        poly = revise_points(poly)
        p0, p1, p2, p3 = fm_scale * poly
        theta = angle(p1, p0)
        for y, x in coords_in_poly:
            point = fm_scale * np.array([x, y], dtype=np.float32)
            # top
            geo_map[y, x, 0] = calculate_point_line_distance(p0, p1, point)
            # right
            geo_map[y, x, 1] = calculate_point_line_distance(p1, p2, point)
            # bottom
            geo_map[y, x, 2] = calculate_point_line_distance(p2, p3, point)
            # left
            geo_map[y, x, 3] = calculate_point_line_distance(p3, p0, point)
            # angle
            geo_map[y, x, 4] = theta

    score_map = score_map.astype(np.float32)
    return score_map, geo_map


def restore_quandrangle(score_map, geo_map, threshold=0.5, fm_scale=4):
    h, w = geo_map.shape[:2]
    coords = np.mgrid[0:fm_scale * h:fm_scale,
                      0:fm_scale * w:fm_scale].transpose(1, 2, 0)  # (h, w, 2)

    exist_coords = coords[np.where(score_map >= threshold)]
    exist_geo_map = geo_map[np.where(score_map >= threshold)]

    p_y, p_x = np.split(exist_coords, 2, axis=1)
    top, right, bottom, left, theta = np.split(exist_geo_map, 5, axis=1)

    top_y = p_y - top
    bot_y = p_y + bottom
    left_x = p_x - left
    right_x = p_x + right

    tl = np.concatenate([left_x, top_y], axis=1)
    tr = np.concatenate([right_x, top_y], axis=1)
    br = np.concatenate([right_x, bot_y], axis=1)

    center_x = np.mean([left_x, right_x], axis=0)
    center_y = np.mean([top_y, bot_y], axis=0)
    center = np.concatenate([center_x, center_y], axis=1)

    shift_tl = tl - center
    shift_tr = tr - center
    shift_bl = bl - center
    shift_br = br - center

    theta = np.squeeze(theta)
    x_rot_matrix = np.stack([np.cos(theta), -np.sin(theta)], axis=1)
    y_rot_matrix = np.stack([np.sin(theta), np.cos(theta)], axis=1)

    rotated_tl_x = (np.sum(x_rot_matrix * shift_tl, axis=1) + center[:, 0])
    rotated_tl_y = (np.sum(y_rot_matrix * shift_tl, axis=1) + center[:, 1])

    rotated_tr_x = (np.sum(x_rot_matrix * shift_tr, axis=1) + center[:, 0])
    rotated_tr_y = (np.sum(y_rot_matrix * shift_tr, axis=1) + center[:, 1])

    rotated_br_x = (np.sum(x_rot_matrix * shift_br, axis=1) + center[:, 0])
    rotated_br_y = (np.sum(y_rot_matrix * shift_br, axis=1) + center[:, 1])

    rotated_bl_x = (np.sum(x_rot_matrix * shift_bl, axis=1) + center[:, 0])
    rotated_bl_y = (np.sum(y_rot_matrix * shift_bl, axis=1) + center[:, 1])

    rotated_tl = np.stack([rotated_tl_x, rotated_tl_y], axis=-1)
    rotated_tr = np.stack([rotated_tr_x, rotated_tr_y], axis=-1)
    rotated_bl = np.stack([rotated_bl_x, rotated_bl_y], axis=-1)
    rotated_br = np.stack([rotated_br_x, rotated_br_y], axis=-1)

    rotated_polys = np.stack([rotated_tl,
                              rotated_tr,
                              rotated_br,
                              rotated_bl], axis=1)

    return rotated_polys
