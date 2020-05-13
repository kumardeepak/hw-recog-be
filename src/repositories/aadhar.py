import cv2
import tensorflow as tf
import os
import time
import numpy as np
import imutils
import pytesseract
from repositories.east_utils import model
from repositories.east_utils import postprocess
from repositories.bbox_tools import Box_cordinates

# margin
# tesseract confidence
# margin

checkpoint_path = '/home/ubuntu/.models/east_icdar2015_resnet_v1_50_rbox/'

tf.reset_default_graph ()
input_images = tf.placeholder (tf.float32, shape=[None, None, None, 3], name='input_images')
global_step = tf.get_variable ('global_step', [], initializer=tf.constant_initializer (0), trainable=False)
f_score, f_geometry = model.model (input_images, is_training=False)
variable_averages = tf.train.ExponentialMovingAverage (0.997, global_step)
saver = tf.train.Saver (variable_averages.variables_to_restore ())
sess = tf.Session (config=tf.ConfigProto (allow_soft_placement=True))
ckpt_state = tf.train.get_checkpoint_state (checkpoint_path)
model_path = os.path.join (checkpoint_path, os.path.basename (ckpt_state.model_checkpoint_path))
print ('Restore from {}'.format (model_path))
saver.restore (sess, model_path)


class Aadhaar_exract:

    def __init__(self, image_path, session=sess):
        self.image    = cv2.imread (image_path)
        self.sess     = session
        self.timer    = {'net': 0, 'restore': 0, 'nms': 0}
        self.text     = {}
        self.extract_text()

    def east_output(self):
        out_put                        = []
        start                          = time.time ()
        im_resized, (ratio_h, ratio_w) = postprocess.resize_image (self.image)
        score, geometry                = self.sess.run ([f_score, f_geometry], feed_dict={input_images: [im_resized]})
        self.timer ['net']             = time.time () - start
        boxes, self.timer              = postprocess.detect (score_map=score, geo_map=geometry, timer=self.timer)
        print (' net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format (self.timer ['net'] * 1000,
                                                                       self.timer ['restore'] * 1000,
                                                                       self.timer ['nms'] * 1000))

        if boxes is not None:
            boxes            = boxes [:, :8].reshape ((-1, 4, 2))
            boxes [:, :, 0] /= ratio_w
            boxes [:, :, 1] /= ratio_h
        #
        if boxes is not None:
            for box in boxes:

                # to avoid submitting errors
                #box = postprocess.sort_poly (box.astype (np.int32))
                print(box)
                if np.linalg.norm (box [0] - box [1]) < 5 or np.linalg.norm (box [3] - box [0]) < 5:
                    continue
                out_put.append (
                    [box [0, 0], box [0, 1], box [1, 0], box [1, 1], box [2, 0], box [2, 1], box [3, 0], box [3, 1]])
        return out_put

    def get_rotaion_angle(self, east_coordinates):

        bboxex                = Box_cordinates (east_coordinates)
        bboxex.df ['delta_x'] = bboxex.df ['x2'] - bboxex.df ['x1']
        bboxex.df ['delta_y'] = bboxex.df ['y2'] - bboxex.df ['y1']
        box_dir               = [bboxex.df ['delta_x'].mean (), bboxex.df ['delta_y'].mean ()]
        # print(box_dir)
        x_axis                = [1, 0]
        cosine                = np.dot (box_dir, x_axis) / (np.linalg.norm (box_dir) * np.linalg.norm (x_axis))
        angle                 = np.arccos (cosine) * 180 / np.pi
        avrage_height         = bboxex.df ['height'].mean ()
        avrage_width          = bboxex.df ['width'].mean ()
        if avrage_height > avrage_width:
            angle = 90 - angle

        return angle * np.sign (box_dir [1])

    def check_orientation(self, group_cordinates, margin=5):
        upside_down = False
        orientation = []
        for index, block in enumerate (group_cordinates):
            crop = self.image [block [0] [1] - margin: block [1] [1] + margin,
                   block [0] [0] - margin: block [1] [0] + margin]
            try:
                osd   = pytesseract.image_to_osd (crop)
                angle = osd.split ('\nRotate') [0].split (': ') [-1]
                orientation.append (int (angle))
            except:
                pass
        orientation     = np.array (orientation)
        chk_orientation = orientation > 170

        # Taking vote of regions
        if chk_orientation.sum () > (len (orientation) * 0.5):
            print ('Image is upside down')
            upside_down = True
            return upside_down

        return upside_down

    def extract_text(self):
        east_cor = self.east_output ()
        angle = self.get_rotaion_angle (east_cor)
        rotations =  1
        # Orientation correction
        while abs (angle) > 2.5:
            self.image = imutils.rotate_bound (self.image, -angle)
            
            if rotations > 1 :
                contours = cv2.findContours (cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = contours [0] if len (contours) == 2 else contours [1]
                if len(contours) > 0 :
                    x, y, w, h = cv2.boundingRect (contours[0])
                    print('cropped area reduced ')
                    self.image = self.image[y:y+h,x:x+w,:]
            east_cor = self.east_output ()
            angle = self.get_rotaion_angle (east_cor)
            rotations += 1
        bbox1 = Box_cordinates (east_cor)
        upside_down = self.check_orientation (bbox1.gr_cordinates)
        if upside_down:
            self.image = imutils.rotate_bound (self.image, 180)
            east_cor = self.east_output ()
        bbox2 = Box_cordinates (east_cor, 50, self.image)
        self.text = bbox2.get_text ()
