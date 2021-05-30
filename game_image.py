import cv2
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import logging

logging.basicConfig(level=logging.INFO)

from collections import namedtuple
import base64
from enum import IntEnum, unique as enum_uniq

########################################################################################################################
# Helper Named tuples and enums
########################################################################################################################
Trigo = namedtuple('Trigo', ('sin', 'cos'))
LineInfo = namedtuple('LineInfo', ('trigo', 'line'))
Border = namedtuple('Border', ('low', 'high'))
HoughParameters = namedtuple('HoughParameters', ('rho', 'theta', 'threshold', 'min_line_length', 'max_line_gap'))

@enum_uniq
class CellExtraEval(IntEnum):
    """ Enum for possible cell evaluations mode
    """
    NO_EVAL = 0  # Image is returned as it saved in cache
    EVAL_BLUR = 1  # Perform Blur operation on the image
    EVAL_BLUR_AND_THRESHOLD = 3  # After blur cut all the pixels with color close to white

########################################################################################################################
# Constants
########################################################################################################################

# Parameters for high level cells detection
SINUS_H = 0.045  # corresponds to ~ 4.5 degree angle
SINUS_V = 0.09  # corresponds to ~ 9 degree angle
FLOAT_ZERO = 0.000001  # sometimes float operations may give non-zero result...
H_LINE_STEP = 10  # maximal distance between two horizontal lines in which those lines considered the same
V_LINE_STEP = 30  # maximal distance between two vertical lines in which those lines considered the same
V_MIN_PIXELS_IN_CELL = 7  # minimal pixels between up and down
H_MIN_PIXELS_IN_CELL = 50  # minimal pixels between left and right
H_HOUGH_PARAMETERS = HoughParameters(rho=1,
                                     theta=np.pi / 720,
                                     threshold=100,
                                     min_line_length=100,
                                     max_line_gap=5)
V_HOUGH_PARAMETERS = HoughParameters(rho=1,
                                     theta=np.pi / 720,
                                     threshold=100,
                                     min_line_length=100,
                                     max_line_gap=5)

# Cell area parametes
UP_ADDITION = 5  # some letters are written a bit above cell, so adding 5 pixels more
DOWN_ADDITION = 5  # some letters are written a bit below cell, so adding 5 pixels more
SIZE_TO_REMOVE_CNT = 10  # Countours with bounding rect less than that will be removed

# Clear border lines parameters
H_LINE_FROM_BORDER = 20  # Maximal distance from border to horizontal line
V_LINE_FROM_BORDER = 30  # Maximal distance from border to vertical line
LEFT_CLEAR_ADDITION = 3  # Clear area between left line+LEFT_CLEAR_ADDITION and border
RIGHT_CLEAR_ADDITION = 3  # Clear area between right line-RIGHT_CLEAR_ADDITION and border
UP_CLEAR_ADDITION = 6  # Clear lines around UP_CLEAR_ADDITION from cell upper line
DOWN_CLEAR_ADDITION = 5  # Clear area between right line-RIGHT_CLEAR_ADDITION and border
H_REMOVE_STEP = 3  # step for remove horizontal lines
H_HOUGH_CELL_PARAMETERS = HoughParameters(rho=1,
                                          theta=np.pi / 720,
                                          threshold=50,
                                          min_line_length=50,
                                          max_line_gap=2)
V_HOUGH_CELL_PARAMETERS = HoughParameters(rho=1,
                                          theta=np.pi / 720,
                                          threshold=25,
                                          min_line_length=25,
                                          max_line_gap=5)

AFTER_BLUR_THRESHOLD = 70  # Threshold after cell bluring

########################################################################################################################
# Game image class
########################################################################################################################
class GameImage:
    """ Class which contains and manipulates (removes noise, divides into cells, etc..) game image
    """
    def __init__(self, img):
        # load image and remove noice
        self.original_image = img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clean = cv2.fastNlMeansDenoising(gray, 7, 7, 21)

        self.binary_image = cv2.adaptiveThreshold(clean, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                                  11, 3)
        """ Monochrome inverted image with values 0 and 255  """

        self.h_borders = []
        """ Borders for horizontal lines (will be initialized after table detection) """
        self.v_borders = []
        """ Borders for v lines (will be initialized after table detection) """
        self.cells = []
        """ 2 dimensional list for cells """
        self.table_columns = []
        """ Indicies of table columns"""
        self.table_rows = []
        """ Indicies of table rows"""

    @classmethod
    def from_filename(cls, filename):
        return GameImage(cv2.imread(filename))

    @classmethod
    def from_uri(cls, uri):
        encoded_data = uri.split(',')[1]
        nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return GameImage(img)

    @classmethod
    def h_value(cls, line, m):
        """ Helper method, returns intersection of horizontal line with x=m

        Parameters
        ----------------
        line : np.array
            line
        m: float
            x value

        Returns
        ----------------
        float
            y
        """
        for x1, y1, x2, y2 in line:
            assert (x1 != x2)
            return y1 + (m - x1) * (y2 - y1) / (x2 - x1)

    @classmethod
    def v_value(cls, line, m):
        """ Helper method, returns intersection of vertical line with y=m

        Parameters
        ----------------
        line : np.array
            line
        m: float
            y value

        Returns
        ----------------
        float
            x
        """
        for x1, y1, x2, y2 in line:
            assert (y1 != y2)
            return x1 + (m - y1) * (x2 - x1) / (y2 - y1)

    @classmethod
    def h_create_lineinfo(cls, line):
        """ Helper method, creates LineInfo from horizontal line

        Parameters
        ----------------
        line : np.array
            line

        Returns
        ----------------
        Optional(LineInfo)
            None if line considered to be non-horizontal, line info otherwise
        """
        for x1, y1, x2, y2 in line:
            norm = cv2.norm(np.float32([[x1 - x2, y1 - y2]]), normType=cv2.NORM_L2)
            if norm != 0:
                if abs(x2 - x1) > abs(y2 - y1):
                    # horisontal line put angle from -pi/4 till pi/4 (cos > 0)
                    sign = 1.0 if (x2 > x1) else -1.0
                    return LineInfo(trigo=Trigo(sin=sign * (y2 - y1) / norm, cos=sign * (x2 - x1) / norm), line=line)
        return None

    @classmethod
    def v_create_lineinfo(cls, line):
        """ Helper method, creates LineInfo from vertical line

        Parameters
        ----------------
        line : np.array
            line

        Returns
        ----------------
        Optional(LineInfo)
            None if line considered to be non-vertical, line info otherwise
        """
        for x1, y1, x2, y2 in line:
            norm = cv2.norm(np.float32([[x1 - x2, y1 - y2]]), normType=cv2.NORM_L2)
            if norm != 0:
                if abs(x2 - x1) < abs(y2 - y1):
                    # vertical line put angle from  rom pi/4 till 3*pi/4
                    sign = 1.0 if (y2 > y1) else -1.0
                    return LineInfo(trigo=Trigo(sin=sign * (y2 - y1) / norm, cos=sign * (x2 - x1) / norm), line=line)
        return None

    @classmethod
    def diff_sinus(cls, lineinfo1, lineinfo2):
        """ Retuns sinus between two lines

        Parameters
        ----------------
        lineinfo1 : LineInfo
            lineinfo of first line
        lineinfo2 : LineInfo
            lineinfo of second line

        Returns
        ----------------
        float
            sinus
        """
        return lineinfo2.trigo.sin * lineinfo1.trigo.cos - lineinfo2.trigo.cos * lineinfo1.trigo.sin

    @classmethod
    def points_to_line(cls, l):
        """ Retuns line according to two points

        Parameters
        ----------------
        l : np.array
            2 dimensional array with two points
        Returns
        ----------------
        np.array
            line
        """
        return np.array([[l[0][0], l[0][1], l[1][0], l[1][1]]])

    def find_parallel_lines(self, max_ignored_sinus, lineinfo_calculate, hough_params, sort_key):
        """ Finds maximal number of parallel (or to be precise close to parallel with MIN_SINUS pressision) lines inside
        given list of lines. Lines are assumed to be in range (-pi/4, pi/4]

        Parameters
        ----------------
        max_ignored_sinus : float
            maximal sinus of angle between to lines considered as parallel
        lineinfo_calculate: Callable
            function returns line info according to line, and None if line is not needed
        hough_params : HoughParameters
            parameters for houghP function
        sort_key:
            key function to sort, insures that angle is increasing - for hosisontal it is sin, for vertical it is -cos

        Returns
        ----------------
        list[LineInfo]
            List of parallel lines
        """

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(self.binary_image,
                                hough_params.rho,
                                hough_params.theta,
                                hough_params.threshold,
                                np.array([]),
                                hough_params.min_line_length,
                                hough_params.max_line_gap)

        line_info_list = [lineinfo_calculate(x) for x in lines if lineinfo_calculate(x) is not None]

        # Entire algorithm is o(n^2) but we want to work with angle ranges and
        # so asin/ln for this array seems to be heavier
        angle_range = (0, 0)
        for i in range(len(line_info_list)):
            line_info_list.sort(key=sort_key)
            j = i + 1
            while j < len(line_info_list):
                # sin(a-b)=sina *cosb - cosa*sinb
                diff_sin = GameImage.diff_sinus(line_info_list[i], line_info_list[j])

                # we have increasing angle in [-pi/2, pi] range diff sin cannot be negative
                assert diff_sin > -FLOAT_ZERO, \
                    f"diff_sin={diff_sin}, i={i}, j={j},{line_info_list[i]},{line_info_list[j]}"

                if diff_sin < max_ignored_sinus:
                    j += 1
                else:
                    break
            if angle_range[1] - angle_range[0] < j - i:
                angle_range = (i, j)
        return line_info_list[angle_range[0]:angle_range[1]]

    def find_cell_borders(self, line_info_list, is_horisontal):
        """ Finds cell borders according to parallel lines

        Parameters
        ----------------
        line_info_list : list[namedtuple]
            list of parallel lines lineinfo tuples
        is_horisontal:
            flag if parallel lines are horisontal
        """

        if is_horisontal:
            image_size = len(self.binary_image[0])
            step = H_LINE_STEP
            self.h_borders.clear()
        else:
            image_size = len(self.binary_image)
            step = V_LINE_STEP
            self.v_borders.clear()

        # Find intersections with the middle line (y=image_size/2 or x=image_size/2) and sort
        middle_intersections = []
        for elem in line_info_list:
            if is_horisontal:
                middle_intersections.append((GameImage.h_value(elem.line, image_size / 2), elem.line))
            else:
                middle_intersections.append((GameImage.v_value(elem.line, image_size / 2), elem.line))

        middle_intersections.sort(key=lambda x: x[0])

        # consider those in the step range to be in the same line, and find up and bottom
        # (or left and right) intersections
        i = 0
        while i < len(middle_intersections):
            j = i
            low_min = high_min = low_max = high_max = None
            while j < len(middle_intersections) and middle_intersections[j][0] <= (middle_intersections[i][0] + step):
                line = middle_intersections[j][1]
                # min/max are needed to take care on lines exiting beyond the image
                if is_horisontal:
                    low = max(0, min(int(GameImage.h_value(line, 0)), len(self.binary_image)))
                    high = max(0, min(int(GameImage.h_value(line, image_size)), len(self.binary_image)))
                else:
                    low = max(0, min(int(GameImage.v_value(line, 0)), len(self.binary_image[0])))
                    high = max(0, min(int(GameImage.v_value(line, image_size)), len(self.binary_image[0])))

                low_min = low if low_min is None or low < low_min else low_min
                low_max = low if low_max is None or low > low_max else low_max
                high_min = high if high_min is None or high < high_min else high_min
                high_max = high if high_max is None or high > high_max else high_max

                j += 1

            if is_horisontal:
                self.h_borders.append(Border(low=np.array([[0, int(low_min), image_size, int(high_min)]]),
                                             high=np.array([[0, int(low_max), image_size, int(high_max)]])))
            else:
                self.v_borders.append(Border(low=np.array([[int(low_min), 0, int(high_min), image_size]]),
                                             high=np.array([[int(low_max), 0, int(high_max), image_size]])))
            i = j

    def find_table(self):
        """ Detects table structure in the image
        """
        h_lines_info = self.find_parallel_lines(SINUS_H,
                                                GameImage.h_create_lineinfo,
                                                H_HOUGH_PARAMETERS,
                                                lambda x: x.trigo.sin)
        v_lines_info = self.find_parallel_lines(SINUS_V,
                                                GameImage.v_create_lineinfo,
                                                V_HOUGH_PARAMETERS,
                                                lambda x: -x.trigo.cos)

        self.find_cell_borders(h_lines_info, True)
        self.find_cell_borders(v_lines_info, False)

    @classmethod
    def removeCountours(cls, im):
        """ Removes countours with dimensions less than threshold (SIZE_TO_REMOVE_CNT).

         Parameters
        ----------------
        im : np.array
            image

        Returns
        ----------------
        np.array
            image
        """
        contours, _ = cv2.findContours(im,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            _, _, w, h = cv2.boundingRect(cnt)
            # _, (w, h), _ = cv2.minAreaRect(cnt)
            if w < SIZE_TO_REMOVE_CNT or h < SIZE_TO_REMOVE_CNT:
                cv2.fillPoly(im, [cnt], 0)

        return im

    def clear_cell_v_borders(self, im, left_border, right_border):
        """ Clears vertical cell borders - her we can find left and right lines and simple clear everything beyound
        them

         Parameters
        ----------------
        im : np.array
            image
        left_border: np.array
            left border line of the image
        right_border: np.array
            left border line of the image

        Returns
        ----------------
        np.array
            image
        """
        im_h = len(im)
        im_w = len(im[0])
        median_h = int(im_h / 2)

        lines = cv2.HoughLinesP(im,
                                V_HOUGH_CELL_PARAMETERS.rho,
                                V_HOUGH_CELL_PARAMETERS.theta,
                                int(im_h / 3),
                                np.array([]),
                                int(im_h / 3),
                                V_HOUGH_CELL_PARAMETERS.max_line_gap)

        if lines is None:
            return im

        # Find vertical lines
        left_border_info = GameImage.v_create_lineinfo(left_border)
        check_line = lambda x: (GameImage.v_create_lineinfo(x) is not None) and \
                               (abs(GameImage.diff_sinus(GameImage.v_create_lineinfo(x), left_border_info)) < SINUS_V)
        v_lines = [x for x in lines if check_line(x)]

        if len(v_lines) == 0:
            # No vertical lines
            return im

        # clear everything left to most left line
        l_min = min(v_lines, key=lambda x: GameImage.v_value(x, median_h))
        if abs(GameImage.v_value(l_min, median_h) - GameImage.v_value(left_border, median_h)) < V_LINE_FROM_BORDER:
            cv2.fillPoly(im, pts=[np.array([[0, 0],
                                            [int(GameImage.v_value(l_min, 0)) + LEFT_CLEAR_ADDITION, 0],
                                            [int(GameImage.v_value(l_min, im_h)) + LEFT_CLEAR_ADDITION, im_h],
                                            [0, im_h]])], color=0)

        # clear everything right to most right line
        l_max = max(v_lines, key=lambda x: GameImage.v_value(x, median_h))
        if abs(GameImage.v_value(l_max, median_h) - GameImage.v_value(right_border, median_h)) < V_LINE_FROM_BORDER:
            cv2.fillPoly(im, pts=[np.array([[im_w, 0],
                                            [int(GameImage.v_value(l_max, 0)) - RIGHT_CLEAR_ADDITION, 0],
                                            [int(GameImage.v_value(l_max, im_h)) - RIGHT_CLEAR_ADDITION, im_h],
                                            [im_w, im_h]])], color=0)

        return im

    def clear_border_h_line(self, im, line_values, addition):
        """ Clears horizontal border, just copies line above/below border to border lines

        Parameters
        ----------------
        im : np.array
            image
        line_values: list
            list of minimal values for all the lines
        addition: int
            clears additional pixels below/above

        Returns
        ----------------
        im
       """
        im_h = len(im)
        im_w = len(im[0])

        for x in range(0, im_w, H_REMOVE_STEP):
            y = round(line_values[x])
            if y < 0 or y >= im_h:
                continue
            if addition > 0:
                y1 = max(0, y - 1)
                y2 = min(im_h - 1, y + addition - 1)
            else:
                y1 = max(0, y + addition + 1)
                y2 = min(im_h - 1, y + 1)
            x1 = max(0, int(x - SIZE_TO_REMOVE_CNT / 2))
            x2 = min(im_w, int(x + SIZE_TO_REMOVE_CNT / 2))
            if np.max(im[y1, x1:x2]) != 0 and np.max(im[y2, x1:x2]) != 0:
                continue
            im[y1:y2 + 1, x] = 0

            # for a in range(min(sign, addition), max(sign, addition)):
            #     l = np.array([[0, left_h + a, im_w, right_h + a]])
            #     for x in range(0, im_w):
            #         y = round(GameImage.h_value(l, x))
            #         if y < 0 or y >= im_h:
            #             continue
            #         c_y1 = y + addition - a
            #         c_y2 = y + 2 * sign - a
            #         n = 0
            #         for i in range(max(0, x - 3), min(im_w - 1, x + 4)):
            #             if c_y1 >= 0 and c_y1 < im_h and im[c_y1][i] != 0:
            #                 n += 1
            #             if c_y2 >= 0 and c_y2 < im_h and im[c_y2][i] != 0:
            #                 n += 1
            #         if n<3:
            #             im[y][x]=0
        return im

    def clear_cell_h_borders(self, im, up_border, down_border):
        """ Clears horizontal cell borders. The process is different from the vertical part. English letter may easily
        cross the border, so we go over found line and copy lines above/below the border

         Parameters
        ----------------
        im : np.array
            image
        left_border: np.array
            left border line of the image
        right_border: np.array
            left border line of the image

        Returns
        ----------------
        im

        """
        im_w = len(im[0])
        median_w = int(im_w / 2)

        lines = cv2.HoughLinesP(im,
                                H_HOUGH_CELL_PARAMETERS.rho,
                                H_HOUGH_CELL_PARAMETERS.theta,
                                int(im_w / 8),
                                np.array([]),
                                int(im_w / 8),
                                H_HOUGH_CELL_PARAMETERS.max_line_gap)

        if lines is None:
            return im

        # Find horizontal lines
        up_border_info = GameImage.h_create_lineinfo(up_border)
        check_line = lambda x: (GameImage.h_create_lineinfo(x) is not None) and \
                               (abs(GameImage.diff_sinus(GameImage.h_create_lineinfo(x), up_border_info)) < SINUS_H)
        h_lines = [l for l in lines if check_line(l)]
        if len(h_lines) == 0:
            # No horizontal lines
            return im

        top_values = [min(GameImage.h_value(l, x) for l in h_lines) for x in range(0, im_w)]
        if abs(top_values[median_w] - GameImage.h_value(up_border, median_w)) < H_LINE_FROM_BORDER:
            im = self.clear_border_h_line(im, top_values, UP_CLEAR_ADDITION)

        bottom_values = [max(GameImage.h_value(l, x) for l in h_lines) for x in range(0, im_w)]
        if abs(bottom_values[median_w] - GameImage.h_value(down_border, median_w)) < H_LINE_FROM_BORDER:
            im = self.clear_border_h_line(im, bottom_values, -DOWN_CLEAR_ADDITION)

        # _,im=cv2.threshold(cv2.blur(im,(5,5)),80,255,cv2.THRESH_BINARY)

        return im

    def build_cell(self, index):
        """ Returns sub-image corresponds to the specific cell

         Parameters
        ----------------
        index : tuple[int,int]
            tuple for (row, column) index (without table_raw/column translation)

        Returns
        ----------------
        Optional(np.array)
            None if column not exists, binary image corresponds to cell otherwise
        """
        if index[0] >= len(self.h_borders) - 1:
            return None
        if index[1] >= len(self.v_borders) - 1:
            return None

        up = self.h_borders[index[0]]
        down = self.h_borders[index[0] + 1]
        left = self.v_borders[index[1]]
        right = self.v_borders[index[1] + 1]

        # Find max borders
        minx = maxx = miny = maxy = 0
        for x1, y1, x2, y2 in up.low:
            miny = min(y1, y2) - UP_ADDITION
        for x1, y1, x2, y2 in down.high:
            maxy = max(y1, y2) + DOWN_ADDITION
        for x1, y1, x2, y2 in left.low:
            minx = min(x1, x2)
        for x1, y1, x2, y2 in right.high:
            maxx = max(x1, x2)

        im = np.copy(self.binary_image[miny:maxy, minx:maxx])
        im_w = maxx - minx
        im_h = maxy - miny

        left_line = np.array(
            [[int(GameImage.v_value(left.low, miny)) - minx, 0],
             [int(GameImage.v_value(left.low, maxy)) - minx, im_h]])
        right_line = np.array(
            [[int(GameImage.v_value(right.low, miny)) - minx, 0],
             [int(GameImage.v_value(right.low, maxy)) - minx, im_h]])

        # Most of letters are going below the line so down line is filtered maximally
        up_line = np.array(
            [[0, int(GameImage.h_value(up.high, minx)) - UP_ADDITION - miny],
             [im_w, int(GameImage.h_value(up.high, maxx)) - UP_ADDITION - miny]])
        down_line = np.array(
            [[0, int(GameImage.h_value(down.high, minx)) + DOWN_ADDITION - miny],
             [im_w, int(GameImage.h_value(down.high, maxx)) + DOWN_ADDITION - miny]])

        # # clear extra area
        cv2.fillPoly(im, pts=[np.array([[0, 0], up_line[0], up_line[1], [im_w, 0]])], color=0)
        cv2.fillPoly(im, pts=[np.array([[0, im_h], down_line[0], down_line[1], [im_w, im_h]])], color=0)
        cv2.fillPoly(im, pts=[np.array([[0, 0], left_line[0], left_line[1], [0, im_h]])], color=0)
        cv2.fillPoly(im, pts=[np.array([[im_w, 0], right_line[0], right_line[1], [im_w, im_h]])], color=0)
        im = self.clear_cell_v_borders(im, GameImage.points_to_line(left_line), GameImage.points_to_line(right_line))
        im = self.clear_cell_h_borders(im, GameImage.points_to_line(up_line), GameImage.points_to_line(down_line))
        im = GameImage.removeCountours(im)
        return im

    def build_table_cells(self, progress_callback):
        """ Builds cells array - binary image for each cell. Array is built according borders (without table_rows or
            columns translation)
        Parameters
        ----------------
        progress_callback: Option[Callable]
            callback to update progress
        """
        self.find_table()
        for i in range(0, len(self.h_borders)):
            row = []
            for j in range(0, len(self.v_borders) - 1):
                row.append(self.build_cell((i, j)))
                if progress_callback is not None:
                    progress_callback((i * len(self.v_borders) + j) / len(self.h_borders) * len(self.v_borders))
            self.cells.append(row)
        self.restore_to_initial()

    def get_binary_cell(self, index):
        """ Returns cell with given index, its user interface, so table_rows/colums translation is performed

         Parameters
        ----------------
        index : tuple[int,int]
            tuple for (row, column) index

        Returns
        ----------------
        Optional(np.array)
            None if column not exists, binary image corresponds to cell otherwise
        """
        if index[0] >= len(self.table_rows) or index[1] >= len(self.table_columns):
            return None
        return self.cells[self.table_rows[index[0]]][self.table_columns[index[1]]]

    def get_colored_cell(self, index, cell_extra_eval = CellExtraEval.NO_EVAL):
        """ Returns cell with given index, its user interface, so table_rows/colums translation is performed

         Parameters
        ----------------
        index : tuple[int,int]
            tuple for (row, column) index

        cell_extra_eval : CellExtraEval
            additional evaluation mode

        Returns
        ----------------
        Optional(np.array)
            None if column not exists, colored image corresponds to cell otherwise
        """
        im=np.copy(self.get_binary_cell(index))
        if cell_extra_eval == CellExtraEval.EVAL_BLUR:
            im=cv2.blur(im, (5, 5))
        if cell_extra_eval == CellExtraEval.EVAL_BLUR_AND_THRESHOLD:
            im=cv2.blur(im, (5, 5))
            _,tmp_binary = cv2.threshold(im,AFTER_BLUR_THRESHOLD,255,cv2.THRESH_BINARY)
            tmp_binary=GameImage.removeCountours(tmp_binary)
            im[tmp_binary == 0]=0
        return cv2.cvtColor(cv2.bitwise_not(im), cv2.COLOR_GRAY2RGB)

    def get_table_rows_number(self):
        """ Returns number of rows in table

        Returns
        ----------------
        int
            Number of rows in table
        """
        return len(self.table_rows)

    def get_table_columns_number(self):
        """ Returns number of columns in table

        Returns
        ----------------
        int
            Number of columns in table
        """
        return len(self.table_columns)

    def remove_row(self, row):
        """ Removes given raw from translation
        """
        del self.table_rows[row]

    def remove_column(self, column):
        """ Removes given column from translation
        """
        del self.table_columns[column]

    def restore_to_initial(self):
        """ Restores rows and columns to initial state - undo all remove row/column operations
        """
        self.table_rows = [x for x in range(0, len(self.h_borders) - 1)]
        self.table_columns = [x for x in range(0, len(self.v_borders) - 1)]

    def debug_show_borders(self):
        """ Draws lines on given image, needed for debugging only
        """
        im = cv2.cvtColor(cv2.bitwise_not(self.binary_image), cv2.COLOR_GRAY2BGR)
        for borders_info in (self.h_borders, self.v_borders):
            for elem in borders_info:
                for x1, y1, x2, y2 in elem.low:
                    cv2.line(im, (x1, y1), (x2, y2), (0, 0, 255), 1)
                for x1, y1, x2, y2 in elem.high:
                    cv2.line(im, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # cv2.imshow("Borders image", im)
        plt.imshow(im)

    def debug_draw_table(self):
        """ Draws cells in table view, used for debugging
        """
        im = None

        for i in range(0, self.get_table_rows_number()):
            row_im = None
            for j in range(0, self.get_table_columns_number()):
                cell = self.get_colored_cell((i, j))
                row_im = np.copy(cell) if row_im is None else np.concatenate((row_im, cell), 1)
                border = np.zeros((len(cell), 4, 3), dtype=np.uint8)
                border[:, :, 0] = 255
                row_im = np.concatenate((row_im, border), 1)
            im = row_im if im is None else np.concatenate((im, row_im), 0)
            border = np.zeros((4, len(im[0]), 3), dtype=np.uint8)
            border[:, :, 0] = 255
            im = np.concatenate((im, border), 0)

        plt.imshow(im)


def test(filename):
    g = GameImage.from_filename(filename)
    g.build_table_cells(None)
    # g.remove_column(0)
    # g.remove_column(2)
    # g.remove_column(4)
    # g.remove_row(0)
    # g.remove_row(0)
    # g.remove_row(0)

    # g.debug_show_borders()
    g.debug_draw_table()

# test("games/1.jpg")
