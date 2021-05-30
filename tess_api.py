from tesserocr import PyTessBaseAPI as TessApi, OEM, PSM, RIL, iterate_level
import cv2
import PIL
import numpy as np
import os
from game_image import GameImage

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from move_ocr_data import DetectedSymbol, ImageOcrData

########################################################################################################################
# Constants
########################################################################################################################
PATH_TO_TESSERACT_ENV = "TESSERACT"
TRAIN_LIST = [("eng","digits_comma"),
              ("eng-best","digits_comma"),
              ("eng_avg","digits_comma"),
              ("eng","digits")
              ]


########################################################################################################################
# Helper Interface classes
########################################################################################################################
class ImagesSupplier:
    """ Interface class for images supplier
    """

    def get_next_image(self):
        """ Returns next image, or None if all images were evaluated

        Returns
        ----------------
        Optional[Any,np.array]]
            tuple (label, image), where label is any unique label of the image, and image - image in cv2 format
        """
        pass


def perform_recognize_cycle(generalApi, digitsApi, pil_image, add_to_results, image):
    """ Build ocr data for image with given api

    Parameters
    ----------------
    generalApi : TessApi
        Tesseract api to detect any symbol
    digitsApi : TessApi
        Tesseract api to detect numbers only
    pil_image: PIL.Image
        image to run engine on it
    add_to_results: Callable
        function to call to add results
    image: Optional[np.array]
        if not None, bounding rect will be drawn on it (used for debugging)

    """

    generalApi.SetImage(pil_image)
    generalApi.Recognize()

    level = RIL.SYMBOL

    # Go over recognition results and build ocr data
    lstm_choices_index = 0
    for symbols_iter in iterate_level(generalApi.GetIterator(), level):
        try:
            symbol = symbols_iter.GetUTF8Text(level)  # r == ri
            rect = symbols_iter.BoundingBox(level)

            digit = None
            digit_prob = {}
            try:
                digitsApi.SetImage(pil_image.crop(rect))
                digitsApi.Recognize()
                for digits_iter in iterate_level(digitsApi.GetIterator(), level):
                    digit = digits_iter.GetUTF8Text(level)
                    ch = digits_iter.GetBestLSTMSymbolChoices()
                    # print(f"Digit {digit}")
                    # for x in ch:
                    #     print(x)
                    # print("\n")
                    if len(ch) > 0:
                        digit_prob = {x[0]: x[1] for x in ch[0]}
                        values_sum = sum([x[1] for x in digit_prob.items()])
                        digit_prob = {x[0]: x[1] / values_sum for x in digit_prob.items()}
                    break
            except (RuntimeError, SystemError):
                print("Digit runtime error")

            ch = symbols_iter.GetBestLSTMSymbolChoices()
            # print(f"Symbol {symbol}")
            # for x in ch:
            #     print(x)
            # print("\n")

            if lstm_choices_index >= len(ch):
                # First symbol after space
                lstm_choices_index = 1

            if lstm_choices_index < len(ch):
                symbols_prob = {x[0]: x[1] for x in ch[lstm_choices_index]}
                values_sum = sum([x[1] for x in symbols_prob.items()])
                symbols_prob = {x[0]: x[1] / values_sum for x in symbols_prob.items()}

                lstm_choices_index += 1
                add_to_results(DetectedSymbol(symbol=symbol,
                                              symbol_prob_dict=symbols_prob,
                                              digit=digit,
                                              digit_prob_dict=digit_prob))

            if image is not None:
                cv2.rectangle(image, np.array([rect[0], rect[1]]), np.array([rect[2], rect[3]]), (0, 0, 255), 1)


        except RuntimeError as e:
            print(f"Runtime error: {e}")


def show_images_in_debug_mode(images_list):
    """ In case of debug mode shows images in plt

    Parameters
    ----------------
    images_list : List[np.array]
        list of images to show
    """
    length = len(images_list)
    if length == 0:
        return
    if length == 1:
        plt.imshow(images_list[0])
    else:
        _, axarr = plt.subplots(nrows=int(length / 2) + 1, ncols=2)
        for i in range(length):
            axarr[int(i / 2)][int(i % 2)].imshow(images_list[i])


def build_moves_ocr_data(imageSupplier, debug_mode=False, train_id=0):
    """ Build ocr data for images supplied by image supplier

    Parameters
    ----------------
    imageSupplier : ImagesSupplier
        object to supply the images
    debug_mode: bool
        if True, shows image with char boxes using cv2.imshow function

    Returns
    ----------------
    Dict[Any, ImageOcrData]
        Map from image label supplied with the image and ocr data built by function
    """
    result_dict = {}
    images_list = []
    path = f"{os.environ[PATH_TO_TESSERACT_ENV]}\\tessdata"


    with TessApi(oem=OEM.LSTM_ONLY, psm=PSM.SINGLE_CHAR, lang=TRAIN_LIST[train_id][0], path=path) as generalApi, \
            TessApi(oem=OEM.LSTM_ONLY, psm=PSM.SINGLE_CHAR, lang=TRAIN_LIST[train_id][1], path=path) as digitsApi:

        generalApi.SetVariable("lstm_choice_mode", "2")
        digitsApi.SetVariable("lstm_choice_mode", "2")

        while True:
            next_image = imageSupplier.get_next_image()
            if next_image is None:
                break

            # Recognize
            label, image = next_image
            pil_image = PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image_data = ImageOcrData()
            perform_recognize_cycle(generalApi,
                                    digitsApi,
                                    pil_image,
                                    image_data.add_symbol,
                                    image if debug_mode else None)

            # in debug mode append images
            if debug_mode:
                images_list.append(image)

            result_dict[label] = image_data

    if debug_mode:
        show_images_in_debug_mode(images_list)
    return result_dict


class TestImageSupplier(ImagesSupplier):
    """ Test supplier receives list of file names and supplies them to the supplier"""

    def __init__(self, filenames):
        self.filenames = filenames
        self.index = [0,0]

    def get_next_image(self):
        if self.index[0] >= len(self.filenames):
            return None
        filename = self.filenames[self.index[0]]
        im = cv2.bitwise_not(cv2.cvtColor(cv2.imread(filename),cv2.COLOR_BGR2GRAY))
        eval=self.index[1]
        if eval == 1:
            im=cv2.blur(im,(5,5))
        if eval == 2:
            im = cv2.blur(im, (5, 5))
            _,tmp_binary = cv2.threshold(im,70,255,cv2.THRESH_BINARY)
            tmp_binary=GameImage.removeCountours(tmp_binary)
            im[tmp_binary == 0]=0
            self.index[0] += 1

        self.index[1]=(self.index[1]+1)%3

        return ((filename,eval), cv2.bitwise_not(cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)))


def test(filenames, debug_mode=False):
    ocr_data = build_moves_ocr_data(TestImageSupplier(filenames), debug_mode)
    for label, ocr in ocr_data.items():
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!! {label}  !!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"{ocr}\n")

    if debug_mode:
        cv2.waitKey(0)

    return ocr_data

# moves=test(["games/game1/m1.jpg","games/game1/m2.jpg","games/game1/m3.jpg","games/game1/m4.jpg"])
# print(moves["games/game1/m1.jpg"].move_rank("a4"))
