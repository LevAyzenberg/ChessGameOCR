import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import base64
import PIL
import io

########################################################################################################################
# Helper Named tuples
########################################################################################################################
Trigo = namedtuple('Trigo', ('sin', 'cos'))
LineInfo = namedtuple('LineInfo', ('trigo', 'line'))
Border = namedtuple('Border', ('low', 'high'))
HoughParameters = namedtuple('HoughParameters', ('rho', 'theta', 'threshold', 'min_line_length', 'max_line_gap'))

########################################################################################################################
# Constants
########################################################################################################################
MAX_GRID_CHANGE_RATIO = 0.8
MIN_GRID_VALUE_W = 110
SINUS_H = 0.045  # corresponds to ~ 5 degree angle
SINUS_V = 0.09  # corresponds to ~ 5 degree angle
FLOAT_ZERO = 0.000001  # sometimes float operations may give non-zero result...
H_LINE_STEP = 10  # maximal distance between two horizontal lines in which those lines considered the same
V_LINE_STEP = 30  # maximal distance between two vertical lines in which those lines considered the same
V_MIN_PIXELS_IN_CELL = 7  # minimal pixels between up and down
H_MIN_PIXELS_IN_CELL = 50  # minimal pixels between left and right

# Parameters for h line detection
H_HOUGH_PARAMETERS = HoughParameters(rho=1,
                                     theta=np.pi / 720,
                                     threshold=100,
                                     min_line_length=100,
                                     max_line_gap=5)
# Parameters for v line detection
V_HOUGH_PARAMETERS = HoughParameters(rho=1,
                                     theta=np.pi / 720,
                                     threshold=100,
                                     min_line_length=100,
                                     max_line_gap=5)


########################################################################################################################
# Main class
########################################################################################################################
class GameImage:
    """ Class which contains and manipulates (removes noise, divides into cells, etc..) game image
    """
    def __init__(self, img):
        # load image and remove noice
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
        img=cv2.imdecode(nparr, cv2.IMREAD_COLOR)
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
                diff_sin = line_info_list[j].trigo.sin * line_info_list[i].trigo.cos - \
                           line_info_list[j].trigo.cos * line_info_list[i].trigo.sin

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
            miny = min(y1, y2)
        for x1, y1, x2, y2 in down.high:
            maxy = max(y1, y2)
        for x1, y1, x2, y2 in left.low:
            minx = min(x1, x2)
        for x1, y1, x2, y2 in right.high:
            maxx = max(x1, x2)

        im = np.copy(self.binary_image[miny:maxy, minx:maxx])
        im_w = maxx - minx
        im_h = maxy - miny
        # Calculate average lines (in im coordinates)
        up_line = np.array(
            [[0, int((GameImage.h_value(up.low, minx) + GameImage.h_value(up.high, minx)) / 2) - miny],
             [im_w, int((GameImage.h_value(up.low, maxx) + GameImage.h_value(up.high, maxx)) / 2) - miny]])
        left_line = np.array(
            [[int((GameImage.v_value(left.low, miny) + GameImage.v_value(left.high, miny)) / 2) - minx, 0],
             [int((GameImage.v_value(left.low, maxy) + GameImage.v_value(left.high, maxy)) / 2) - minx, im_h]])
        right_line = np.array(
            [[int((GameImage.v_value(right.low, miny) + GameImage.v_value(right.high, miny)) / 2) - minx, 0],
             [int((GameImage.v_value(right.low, maxy) + GameImage.v_value(right.high, maxy)) / 2) - minx, im_h]])
        # Most of letters are going below the line so down line is filtered maximally
        down_line = np.array(
            [[0, int(GameImage.h_value(down.high, minx)) - miny],
             [im_w, int(GameImage.h_value(down.high, maxx)) - miny]])

        # clear extra area
        cv2.fillPoly(im, pts=[np.array([[0, 0], up_line[0], up_line[1], [im_w, 0]])], color=0)
        cv2.fillPoly(im, pts=[np.array([[0, im_h], down_line[0], down_line[1], [im_w, im_h]])], color=0)
        cv2.fillPoly(im, pts=[np.array([[0, 0], left_line[0], left_line[1], [0, im_h]])], color=0)
        cv2.fillPoly(im, pts=[np.array([[im_w, 0], right_line[0], right_line[1], [im_w, im_h]])], color=0)

        return im

    def build_table_cells(self):
        """ Builds cells array - binary image for each cell. Array is built according borders (without table_rows or
            columns translation)
        """
        self.find_table()
        for i in range(0, len(self.h_borders)):
            row = []
            for j in range(0, len(self.v_borders) - 1):
                row.append(self.build_cell((i, j)))
            self.cells.append(row)
        self.table_rows = [x for x in range(0, len(self.h_borders) - 1)]
        self.table_columns = [x for x in range(0, len(self.v_borders) - 1)]

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

    def get_colored_cell(self, index):
        """ Returns cell with given index, its user interface, so table_rows/colums translation is performed

         Parameters
        ----------------
        index : tuple[int,int]
            tuple for (row, column) index

        Returns
        ----------------
        Optional(np.array)
            None if column not exists, colored image corresponds to cell otherwise
        """
        return cv2.cvtColor(cv2.bitwise_not(self.get_binary_cell(index)),cv2.COLOR_GRAY2RGB)

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

        #cv2.imshow("Borders image", im)
        # plt.imshow(im)

    def debug_draw_table(self):
        """ Draws cells in table view, used for debugging
        """
        im = None
        for i in range(0, self.get_table_rows_number()):
            column_im = None
            for j in range(0, self.get_table_columns_number()):
                cell = self.get_colored_cell((i,j))
                column_im = cell if column_im is None else np.concatenate((column_im, cell), 1)
                border = np.zeros((len(cell), 4, 3))
                border[:, :, 0] = 255
                column_im = np.concatenate((column_im, border), 1)
            im = column_im if im is None else np.concatenate((im, column_im), 0)
            border = np.zeros((4, len(im[0]), 3))
            border[:, :, 0] = 255
            im = np.concatenate((im, border), 0)

        #plt.imshow(im)
        cv2.imwrite("test_table.jpg",im)


def test(filename):
    g = GameImage(filename)
    g.build_table_cells()
    g.remove_column(0)
    g.remove_column(2)
    g.remove_column(4)
    g.remove_row(0)
    g.remove_row(0)
    g.remove_row(0)

    #g.debug_show_borders()
    g.debug_draw_table()

# test("games/1.jpg")
# cv2.waitKey()
