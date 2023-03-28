import cv2
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from mtcnn.mtcnn import MTCNN
import numpy as np

def draw_image_with_boxes(filename, result_list):
    data = pyplot.imread(filename)
    pyplot.imshow(data)
    ax = pyplot.gca()
    for result in result_list:
        x, y, width, height = result['box']
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        ax.add_patch(rect)
        delta_x = result['keypoints']['right_eye'][0] - result['keypoints']['left_eye'][0]
        delta_y = result['keypoints']['right_eye'][1] - result['keypoints']['left_eye'][1]
        angle = np.arctan(delta_y / delta_x)
        delta_l = np.tan(angle)*(abs(abs(result['keypoints']['left_eye'][1]-y)-height))
        for key, value in result['keypoints'].items():
            if key in ['left_eye', 'right_eye', 'nose']:
                line1 = Line2D([value[0]+delta_l,value[0]-delta_l],[y,y+height], color="k")
                ax.add_line(line1)
        print("Расстояние локальных от центра:", abs(result['keypoints']['left_eye'][0] - result['keypoints']['nose'][0]),';',abs(result['keypoints']['right_eye'][0] - result['keypoints']['nose'][0]))
    pyplot.show()
filename = 'img/img1.jpg'
pixels = cv2.imread(filename)
detector = MTCNN()
faces = detector.detect_faces(pixels)
draw_image_with_boxes(filename, faces)
