import cv2
import numpy as np
from cv2 import aruco
from matplotlib import pyplot as plt
from tqdm import tqdm

ARUCO_DICT = cv2.aruco.DICT_5X5_100
MARKER_LENGTH = 0.07
length = 12

PARAMS = aruco.DetectorParameters()
dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
detector = aruco.ArucoDetector(dictionary=dictionary, detectorParams=PARAMS)

video_path = r"data\2_pose_estimation.mp4"

calib_file = np.load(r'calibration_results.npz')
mtx = calib_file['mtx']
dist = np.array([0., 0., 0., 0., 0.])

cap = cv2.VideoCapture(video_path)

total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

mtx[0, 2] = width / 2
mtx[1, 2] = height / 2

# зміна фокусу
# mtx[0, 0] -= 200
# mtx[1, 1] -= 200

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(fr'res/out_video_base.mp4', fourcc, fps, (width, height))


def get_max_distant_corners(tvecs):
    distances = []
    for i in range(len(tvecs)):
        for j in range(i + 1, len(tvecs)):
            dist = np.linalg.norm(tvecs[i, 0] - tvecs[j, 0])
            distances.append((dist, i, j))

    max_dist, idx1, idx2 = max(distances)
    return idx1, idx2


def draw_cube(frame, rvec, tvec, mtx, dist):
    idx1, idx2 = get_max_distant_corners(tvec)
    obj_center = np.mean([tvec[idx1], tvec[idx2]], axis=0)[0]
    obj_orientation = rvec[0]

    rotation_matrix, _ = cv2.Rodrigues(obj_orientation)

    O_X = 0.27
    O_Y = 0.26
    O_Z = 0.12

    object_points = np.array([
        [-O_X / 2, -O_Y / 2, 0],
        [O_X / 2, -O_Y / 2, 0],
        [O_X / 2, O_Y / 2, 0],
        [-O_X / 2, O_Y / 2, 0],
        [-O_X / 2, -O_Y / 2, O_Z],
        [O_X / 2, -O_Y / 2, O_Z],
        [O_X / 2, O_Y / 2, O_Z],
        [-O_X / 2, O_Y / 2, O_Z]
    ])

    image_points = cv2.projectPoints(
        object_points,
        obj_orientation,
        obj_center,
        mtx,
        dist
    )[0].reshape(-1, 2)
    image_points = image_points.reshape(-1, 2).astype(int)
    vis_img = frame.copy()

    for i in range(4):
        cv2.line(vis_img, image_points[i], image_points[i + 4], (0, 255, 0), 2)

    for i1, i2 in zip([0, 1, 2, 3], [1, 2, 3, 0]):
        cv2.line(vis_img, image_points[i1], image_points[i2], (255, 0, 0), 2)
        cv2.line(vis_img, image_points[i1 + 4], image_points[i2 + 4], (0, 0, 255), 2)

    return vis_img


def main():
    process_bar = tqdm(total=total_frames, desc=f'Loading', position=0)

    while True:

        ret, frame = cap.read()
        process_bar.update(1)
        if not ret:
            break

        undist = cv2.undistort(frame, mtx, dist, None, mtx)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        marker_corners, marker_ids, _ = detector.detectMarkers(gray)

        if len(marker_ids) > 2:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(marker_corners, MARKER_LENGTH, mtx, dist)
            aruco.drawDetectedMarkers(gray, marker_corners, marker_ids)
            for i in range(len(marker_corners)):
                for i in range(len(marker_ids)):
                    cv2.drawFrameAxes(undist, mtx, dist, rvecs[i], tvecs[i], 0.03)

            undist = draw_cube(undist, rvecs, tvecs, mtx, dist)

        out.write(undist)

    cap.release()
    out.release()


if __name__ == "__main__":
    main()
