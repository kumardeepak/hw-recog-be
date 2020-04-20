import cv2
import numpy as np


# import os
# import matplotlib.pyplot as plt

class Table_extractor ():

    def __init__(self, filepath, SORT_METHOD='top-to-bottom', MAX_THRESHOLD_VALUE=255, BLOCK_SIZE=15,
                 THRESHOLD_CONSTANT=0, SCALE=15):

        '''

        :param filepath: absolute path of input image file
        :param SORT_METHOD: order of indexing of cells in a table
        :param BLOCK_SIZE: size of neighbourhood taken in account for calculating adaptive threshold
        :param THRESHOLD_CONSTANT: offset used for adaptive thresholding
        :param SCALE: size of pattern finding kernel (line elements in this case)
        '''

        self.image_path = filepath
        self.response = {"response": {"tables": []}}
        self.load_image ()
        self.MAX_THRESHOLD_VALUE = MAX_THRESHOLD_VALUE
        self.BLOCK_SIZE = BLOCK_SIZE
        self.THRESHOLD_CONSTANT = THRESHOLD_CONSTANT
        self.SCALE = SCALE
        self.SORT_METHOD = SORT_METHOD
        self.get_table_mask ()
        self.table_indexing ()

    def load_image(self):
        self.input_image = cv2.imread (self.image_path, 0)
        self.slate = np.zeros (self.input_image.shape)

    def get_table_mask(self):
        # binarization of image
        filtered = cv2.adaptiveThreshold (~self.input_image, self.MAX_THRESHOLD_VALUE, cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, self.BLOCK_SIZE, self.THRESHOLD_CONSTANT)
        # Finding srtuctre elements (horizontal and vertical lines)
        horizontal = filtered.copy ()
        vertical = filtered.copy ()

        horizontal_size = int (horizontal.shape [1] / self.SCALE)
        horizontal_structure = cv2.getStructuringElement (cv2.MORPH_RECT, (horizontal_size, 1))
        horizontal = cv2.erode (horizontal, horizontal_structure)
        horizontal = cv2.dilate (horizontal, horizontal_structure)

        vertical_size = int (vertical.shape [0] / self.SCALE)
        vertical_structure = cv2.getStructuringElement (cv2.MORPH_RECT, (1, vertical_size))
        vertical = cv2.erode (vertical, vertical_structure)
        vertical = cv2.dilate (vertical, vertical_structure)

        # generating table borders
        self.mask = horizontal + vertical
        self.intersections = cv2.bitwise_and (horizontal, vertical)

    def verify_table(self):
        MIN_TABLE_AREA = 50  # min table area to be considered a table
        EPSILON = 3  # epsilon value for contour approximation
        area = cv2.contourArea (contour)

        if (area < MIN_TABLE_AREA):
            return (None, None)

        # approxPolyDP approximates a polygonal curve within the specified precision
        curve = cv2.approxPolyDP (contour, EPSILON, True)

        # boundingRect calculates the bounding rectangle of a point set (eg. a curve)
        rect = cv2.boundingRect (curve)  # format of each rect: x, y, w, h

        # Finds the number of joints in each region of interest (ROI)
        # Format is in row-column order (as finding the ROI involves numpy arrays)
        # format: image_mat[rect.y: rect.y + rect.h, rect.x: rect.x + rect.w]
        possible_table_region = intersections [rect [1]:rect [1] + rect [3], rect [0]:rect [0] + rect [2]]
        (possible_table_joints, _) = cv2.findContours (possible_table_region, cv2.RETR_CCOMP,
                                                       cv2.CHAIN_APPROX_SIMPLE)

        # Determines the number of table joints in the image
        # If less than 5 table joints, then the image
        # is likely not a table
        if len (possible_table_joints) < 5:
            return (None, None)

    def sort_contours(self, cnts, method="left-to-right"):
        reverse = False
        i = 0
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
        boundingBoxes = [cv2.boundingRect (c) for c in cnts]
        (cnts, boundingBoxes) = zip (*sorted (zip (cnts, boundingBoxes), key=lambda b: b [1] [i], reverse=reverse))
        return (cnts, boundingBoxes)

    def draw_contours_index(self, contours, img):
        '''

        :param contours:  contours present cropped fraction of mask image
        :param img: cropped portion of mask image having one table (in case when input image has multiple tables )
        :return: image indexed with cell location, list of bounding box coordinates of every individual cell
        '''
        image_area = img.shape [0] * img.shape [1]
        draw_conts = np.zeros (img.shape)
        # margin = 10
        midpoints = []
        rects = []
        xi, yi = 0, 0
        for i in range (len (contours)):
            cont_area = cv2.contourArea (contours [i])
            x1, y1, w1, h1 = cv2.boundingRect (contours [i])

            #
            if cont_area / float (image_area) < 0.9:
                midpoint = [int (x1 + w1 / 2), int (y1 + h1 / 2)]  # np.mean(contours[i],axis=0)
                midpoints.append (midpoint)
                if len (midpoints) > 1:
                    shift = midpoints [-1] [0] - midpoints [-2] [0]

                    # Detecting change in column by measuring difference in x coordinate of current and previous cell
                    # (cells already sored based on their coordinates)
                    if shift < 10:
                        # print(shift , midpoints[-1][0] , midpoints[-2][0] ,'True')
                        # print(xi,yi)
                        yi = yi + 1
                    else:
                        yi = 0
                        xi = xi + 1
                        # print('False' ,xi,yi)
                rects.append ({"x": x1, "y": y1, "w": w1, "h": h1, "index": (xi, yi)})
                cv2.rectangle (draw_conts, (x1, y1), (x1 + w1, y1 + h1), 255, 1)
                cv2.putText (draw_conts, str ((xi, yi)), (int (midpoint [0]), int (midpoint [1])),
                             cv2.FONT_HERSHEY_SIMPLEX,
                             0.3, 255, 1, cv2.LINE_AA)
        return draw_conts, rects

    def table_indexing(self):

        # list_of_tables = []
        image_area = float (self.input_image.shape [0] * self.input_image.shape [1])

        # finding all the tables in the image, cv2.RETR_EXTERNAL gives only the outermost border of an
        # enclosed figure.
        contours = cv2.findContours (self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours [0] if len (contours) == 2 else contours [1]

        if len (contours) > 0:
            # Indexing one table at a time
            for c in contours:
                x, y, w, h = cv2.boundingRect (c)
                area_ratio = (w * h) / image_area

                # Filtering for noise
                if (area_ratio < 0.8) & (area_ratio > 0.05):
                    table_dic = {"x": x, "y": y, "w": w, "h": h}

                    print (x, y, w, h, area_ratio)
                    crop_fraction = self.mask [y - 2: y + h + 2, x - 2:x + w + 2]

                    sub_contours = cv2.findContours (crop_fraction, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    sub_contours = sub_contours [0] if len (sub_contours) == 2 else sub_contours [1]
                    sorted_conts = sub_contours  # self.sort_contours(sub_contours,method = self.SORT_METHOD)

                    indexed_sub_image, rects = self.draw_contours_index (sorted_conts, img=crop_fraction)
                    table_dic ['rect'] = rects

                    # self.slate stores an image indexed with cell location for all available tables
                    self.slate [y - 2: y + h + 2, x - 2:x + w + 2] = indexed_sub_image

                    self.response ["response"] ["tables"].append (table_dic)