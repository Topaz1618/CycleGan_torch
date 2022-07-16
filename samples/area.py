def polygon_area(polygon):
    """
    compute polygon area
    polygon: list with shape [n, 2], n is the number of polygon points
    """
    area = 0
    q = polygon[-1]
    for p in polygon:
        area += p[0] * q[1] - p[1] * q[0]
        q = p
    return abs(area) / 2.0
