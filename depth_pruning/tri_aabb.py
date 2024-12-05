#https://bronsonzgeb.com/index.php/2021/05/29/gpu-mesh-voxelizer-part-2/''
#5.12.24

import numpy as np

def _intersects_triangle_aabb_sat(v0, v1, v2, aabb_extents, axis):
    """
    Check if a triangle intersects with an AABB along a given axis using the SAT method.
    """
    p0 = np.dot(v0, axis)
    p1 = np.dot(v1, axis)
    p2 = np.dot(v2, axis)

    r = (
        aabb_extents[0] * abs(np.dot(np.array([1, 0, 0]), axis)) +
        aabb_extents[1] * abs(np.dot(np.array([0, 1, 0]), axis)) +
        aabb_extents[2] * abs(np.dot(np.array([0, 0, 1]), axis))
    )

    max_p = max(p0, p1, p2)
    min_p = min(p0, p1, p2)

    return not (max(-max_p, min_p) > r)


def intersects_triangle_aabb(tri_v0, tri_v1, tri_v2, aabb_center:np.ndarray, aabb_extents:np.ndarray):
    """
    Check if a triangle intersects with an AABB.
    triangle: dict with keys 'a', 'b', 'c' representing vertices.
    aabb: dict with keys 'center' and 'extents'.
    """
    tri_a = tri_v0 - aabb_center
    tri_b = tri_v1 - aabb_center
    tri_c = tri_v2 - aabb_center

    ab = tri_b - tri_a
    bc = tri_c - tri_b
    ca = tri_a - tri_c

    # Cross ab, bc, and ca with axes
    axes = []
    axes.append(np.array([0.0, -ab[2], ab[1]]))  # Cross with (1, 0, 0)
    axes.append(np.array([0.0, -bc[2], bc[1]]))
    axes.append(np.array([0.0, -ca[2], ca[1]]))
    axes.append(np.array([ab[2], 0.0, -ab[0]]))  # Cross with (0, 1, 0)
    axes.append(np.array([bc[2], 0.0, -bc[0]]))
    axes.append(np.array([ca[2], 0.0, -ca[0]]))
    axes.append(np.array([-ab[1], ab[0], 0.0]))  # Cross with (0, 0, 1)
    axes.append(np.array([-bc[1], bc[0], 0.0]))
    axes.append(np.array([-ca[1], ca[0], 0.0]))

    # Add box axes
    axes.append(np.array([1, 0, 0]))
    axes.append(np.array([0, 1, 0]))
    axes.append(np.array([0, 0, 1]))

    # Add triangle normal
    tri_normal = np.cross(ab, bc)
    axes.append(tri_normal)

    for axis in axes:
        if not _intersects_triangle_aabb_sat(tri_a, tri_b, tri_c, aabb_extents, axis):
            return False

    return True

# triangle = [
#     np.array([1.0, 1.0, 1.0]),
#     np.array([-1.0, 1.0, 1.0]),
#     np.array([0.0, -1.0, 1.0])
# ]

# aabb = [
#     np.array([0.0, 0.0, 0.0]),
#     np.array([1.0, 1.0, 1.0])
# ]

# print(intersects_triangle_aabb(*triangle, *aabb))