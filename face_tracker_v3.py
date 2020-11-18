import cv2
import dlib
import numpy as np
from imutils import face_utils
import socket
import time

# Predictor model for dlib frontal face detector
p = "shape_predictor_68_face_landmarks.dat"

# 3D model of key facial landmarks for solvePNP
model_points = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corne
    (-150.0, -150.0, -125.0),  # Left Mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner
], dtype=np.float32)

dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

left = [36, 37, 38, 39, 40, 41] # keypoint indices for left eye
right = [42, 43, 44, 45, 46, 47] # keypoint indices for right eye
kernel = np.ones((9, 9), np.uint8)

width = 800
height = 600

DEFAULT_ROTATION = 200
DEFAULT_TRANSLATION = 3000

def contouring(thresh, mid, img, right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
    except:
        pass

class FTC():
    '''
    Face Tracker Class: Holds attributes and functions related to face tracking
    '''

    def __init__(self):
        # Face Recognition Attributes
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(p)
        # WebCam Attributes

        self.cap = None
        self.init_camera()
        self.ret, self.image = self.cap.read()
        self.thresh = self.image.copy()
        self.size = self.image.shape
        # Camera Attributes
        self.focal_length = self.size[1]
        self.center = (self.size[1] / 2, self.size[0] / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0.0, self.center[0]],
             [0.0, self.focal_length, self.center[1]],
             [0.0, 0.0, 1.0]], dtype=np.float32
        )
        self.camera_matrix = np.array(
            [[height, 0.0, width/2],
             [0.0, height, width/2],
             [0.0, 0.0, 1.0]], dtype=np.float32
        )

    def init_camera(self):
        if self.cap == None:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            time.sleep(0.5)

    def value_smooth(self, name, length, value):
        if not hasattr(self, 'smooth'):
            self.smooth = {}
        if not name in self.smooth:
            self.smooth[name] = np.array([value])
        else:
            self.smooth[name] = np.insert(arr=self.smooth[name], obj=0, values=value)
            if self.smooth[name].size > length:
                self.smooth[name] = np.delete(self.smooth[name], self.smooth[name].size-1, 0)
        return np.sum(self.smooth[name])/ self.smooth[name].size

    def limit_rotation(self, rotation):
        '''
        Limits rotation max and min while also fixing the number of digits.
        '''
        rot_lim = rotation
        if rotation > 100:
            rot_lim = 100
        elif rotation < -100:
            rot_lim = -100
        rot_lim += DEFAULT_ROTATION
        return rot_lim

    def limit_translation(self, translation):
        '''
        Limits translation max and min while also fixing the number of digits.
        '''
        trans_limit = translation
        if translation > 2000:
            trans_limit = 2000
        if translation < -2000:
            trans_limit = -2000
        trans_limit += DEFAULT_TRANSLATION
        return trans_limit

    def eye_on_mask(self, mask, shape, side):
        points = [shape[i] for i in side]
        points = np.array(points, dtype=np.int32)
        mask = cv2.fillConvexPoly(mask, points, 255)
        return mask

    def execute(self):
        ret, image = self.cap.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        face_vals = "{};{};{};{};{};{}".format(DEFAULT_ROTATION,
                                               DEFAULT_ROTATION,
                                               DEFAULT_ROTATION,
                                               DEFAULT_TRANSLATION,
                                               DEFAULT_TRANSLATION,
                                               DEFAULT_TRANSLATION)
        if len(rects) == 0:
            return face_vals
        for (i, rect) in enumerate(rects):
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Eye mask over top of image
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask = self.eye_on_mask(mask, shape, left)
            mask = self.eye_on_mask(mask, shape, right)
            mask = cv2.dilate(mask, kernel, 5)
            eyes = cv2.bitwise_and(image, image, mask=mask)
            mask = (eyes == [0, 0, 0]).all(axis=2)
            eyes[mask] = [255, 255, 255]
            mid = (shape[42][0] + shape[39][0]) // 2
            eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
            threshold = 100
            _, thresh = cv2.threshold(eyes_gray, threshold, 255,
                                      cv2.THRESH_BINARY)
            thresh = cv2.erode(thresh, None, iterations=2)  # 1
            thresh = cv2.dilate(thresh, None, iterations=4)  # 2
            thresh = cv2.medianBlur(thresh, 3)  # 3
            thresh = cv2.bitwise_not(thresh)
            contouring(thresh[:, 0:mid], mid, image)
            contouring(thresh[:, mid:], mid, image, True)

            # Head pose direction
            image_points = np.array([shape[30],  # Nose tip - 31
                                     shape[8],  # Chin - 9
                                     shape[36],  # Left eye left corner - 37
                                     shape[45],  # Right eye right corne - 46
                                     shape[48],  # Left Mouth corner - 49
                                     shape[54]  # Right mouth corner - 55
                                     ], dtype=np.float32)
            if hasattr(self, 'rotation_vector'):
                (success, self.rotation_vector, self.translation_vector) = cv2.solvePnP(
                    model_points, image_points, self.camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE, rvec=self.rotation_vector,
                    tvec=self.translation_vector, useExtrinsicGuess=True)
            else:
                (success, self.rotation_vector, self.translation_vector) = cv2.solvePnP(
                    model_points, image_points, self.camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE, useExtrinsicGuess=False)
            if not hasattr(self, 'rotation_vector_initial'):
                self.rotation_vector_initial = np.copy(self.rotation_vector)
            if not hasattr(self, 'translation_vector_initial'):
                self.translation_vector_initial = np.copy(self.translation_vector)
            rx = self.value_smooth("r_x", 5, (
                        self.rotation_vector[0] - self.rotation_vector_initial[0]))
            ry = self.value_smooth("r_y", 5, (
                        self.rotation_vector[1] - self.rotation_vector_initial[1]))
            rz = self.value_smooth("r_z", 5, (
                    self.rotation_vector[2] - self.rotation_vector_initial[2]))
            tx = self.value_smooth("t_x", 4, (
                    self.translation_vector[0] - self.translation_vector_initial[0]))
            ty = self.value_smooth("t_y", 4, (
                    self.translation_vector[1] - self.translation_vector_initial[1]))
            tz = self.value_smooth("t_z", 4, (
                    self.translation_vector[2] - self.translation_vector_initial[2]))
            face_vals = "{};{};{};{};{};{}".format(
                self.limit_rotation(int(rx * 100)),
                self.limit_rotation(int(ry * 100)),
                self.limit_rotation(int(rz * 100)),
                self.limit_translation(int(tx)),
                self.limit_translation(int(ty)),
                self.limit_translation(int(tz))
                )
            # Face Rect
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            for (x, y) in shape:
                cv2.circle(image, (x, y), 2, (255, 0, 0), -1)
        cv2.imshow("image", image)
        return face_vals

    def destroy(self):
        cv2.destroyAllWindows()
        self.cap.release()


if __name__ == "__main__":
    model = FTC()
    while True:
        print(model.execute().encode().__sizeof__())
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    model.destroy()
