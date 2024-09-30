import numpy as np

PARENT_DIR = "jackknife\\templates\\"
# Key-gesture associations
GESTURE_TYPES = {'1':"zigzag", '2':"triangle", '3':"rectangle", '4':"x", 
                 '5':"c", '6':"arrow"}
TEMPLATES_PER_GESTURE = 3
NUM_RESAMPLE_POINTS = 16

NUM_GESTURES = len(GESTURE_TYPES)
# Set Sakoe-Chiba band radius to 10% of resampled time series length
R = int(np.ceil(0.1 * NUM_RESAMPLE_POINTS))


# Resample to NUM_RESAMPLE_POINTS equidistant points along gesture path
def resample(points):
    points_copy = points.copy()
    resampled_points = [] 
    resampled_points.append(points_copy[0])

    point_spacing = path_len(points_copy) / (NUM_RESAMPLE_POINTS - 1)

    # Used in case dist between curr point and last point < point spacing
    accumulated_dist = 0

    i = 1
    while i < len(points_copy) and point_spacing > 0:
        curr_point = np.array(points_copy[i])
        last_point = np.array(points_copy[i - 1])
        curr_dist = np.linalg.norm(curr_point - last_point)

        if accumulated_dist + curr_dist >= point_spacing:
            curr_diff_vec = curr_point - last_point
            if curr_dist != 0:
                next_pnt_factor = (point_spacing - accumulated_dist) / curr_dist
            else:
                next_pnt_factor = 0.5

            resampled_point = points_copy[i - 1] + next_pnt_factor * curr_diff_vec
            resampled_points.append(resampled_point)
            points_copy.insert(i, resampled_point)
            accumulated_dist = 0
        else:   
            accumulated_dist += curr_dist

        i += 1

    while len(resampled_points) < NUM_RESAMPLE_POINTS:
        resampled_points.append(points_copy[-1])

    return resampled_points


def path_len(points):
    length = 0

    for i in range(1, len(points)):
        curr_point = np.array(points[i])
        last_point = np.array(points[i - 1])
        length += np.linalg.norm(curr_point - last_point)

    return length


# Convert n points to n-1 gesture path direction (unit) vectors
def to_gpdvs(points):
    # Convert to numpy array for ease of vector operations
    np_points = np.array(points)
    gpdvs = []

    for i in range(len(np_points) - 1):
        diff_vec = np_points[i + 1] - np_points[i]
        diff_vec_norm = np.linalg.norm(diff_vec)

        # Handles division by 0
        if diff_vec_norm != 0:
            # Normalize
            gpdv = diff_vec / diff_vec_norm
        else:
            gpdv = diff_vec

        gpdvs.append(gpdv)

    return gpdvs