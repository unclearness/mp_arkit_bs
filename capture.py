# https://developers-jp.googleblog.com/2023/07/mediapipe-enhancing-virtual-humans-to-be-more-realistic.html

import os
import cv2
from pathlib import Path
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.framework.formats import landmark_pb2
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# https://github.com/google/mediapipe/issues/4210
MP_TASK_FILE_URL = "https://storage.googleapis.com/mediapipe-assets/face_landmarker_with_blendshapes.task"
MP_TASK_FILE_NAME = "face_landmarker_with_blendshapes.task"
MP_TASK_FILE_PATH = Path(__file__).parent.joinpath("data", MP_TASK_FILE_NAME)


class FaceMeshDetector:
    def __init__(self):
        with open(MP_TASK_FILE_PATH, mode="rb") as f:
            f_buffer = f.read()
        base_options = mp_python.BaseOptions(model_asset_buffer=f_buffer)
        options = mp_python.vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_faces=1,
            result_callback=self.mp_callback)
        self.model = mp_python.vision.FaceLandmarker.create_from_options(
            options)
        self.landmarks = None
        self.blendshapes = None
        self.latest_time_ms = 0

    def mp_callback(self, mp_result, output_image, timestamp_ms: int):
        if len(mp_result.face_landmarks) >= 1 and len(
                mp_result.face_blendshapes) >= 1:
            print(mp_result)
            self.landmarks = mp_result.face_landmarks[0]
            self.blendshapes = [(b.category_name, b.score)
                                for b in mp_result.face_blendshapes[0]][1:]  # 0 is _neutral

    def update(self, frame):
        t_ms = int(time.time() * 1000)
        if t_ms <= self.latest_time_ms:
            return
        frame_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.model.detect_async(frame_mp, t_ms)
        self.latest_time_ms = t_ms

    def get_results(self):
        return self.landmarks, self.blendshapes


def on_trackbar():
    pass


def draw_livestream_landmarks(frame, landmarks):
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y,
                                         z=landmark.z)
        for landmark in landmarks])
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks_proto,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_tesselation_style())
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks_proto,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks_proto,
        connections=mp_face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_iris_connections_style())


is_first = True


def show_blendshape_trackbars(window_title, blendshapes,):
    global is_first
    if is_first:
        for k, v in blendshapes:
            cv2.createTrackbar(
                k,
                window_title,
                0,
                255,
                on_trackbar
            )
        is_first = False
    else:
        for k, v in blendshapes:
            cv2.setTrackbarPos(
                k,
                window_title,
                int(v * 255)
            )


def main():
    if not MP_TASK_FILE_PATH.exists():
        import requests
        urlData = requests.get(MP_TASK_FILE_URL).content
        os.makedirs(MP_TASK_FILE_PATH.parent, exist_ok=True)
        with open(MP_TASK_FILE_PATH, mode='wb') as fp:
            fp.write(urlData)

    facemesh_detector = FaceMeshDetector()
    cap = cv2.VideoCapture(0)
    window_title = "mediapipe ARKit 52 blendshapes"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        facemesh_detector.update(frame)
        landmarks, blendshapes = facemesh_detector.get_results()
        if (landmarks is None) or (blendshapes is None):
            continue
        show_blendshape_trackbars(window_title, blendshapes)
        draw_livestream_landmarks(frame, landmarks)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
