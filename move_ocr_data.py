from collections import namedtuple
import math

########################################################################################################################
# Helper Named tuples
########################################################################################################################
DetectedSymbol = namedtuple("DetectedSymbol", ("symbol", "symbol_prob_dict", "digit", "digit_prob_dict"))
""" Symbol information supplied by ocr engine """

########################################################################################################################
# Constants
########################################################################################################################

# mapping of symbols (engine detects more symbols than we need and part of them are close enough to what we need)
GENERAL_SYMBOLS_MAPPING = {'M': 'N',
                           'W': 'N',
                           'C': 'c',
                           '¢': 'c',
                           '<': 'c',
                           'r': 'c',
                           '@': 'e',
                           'F': 'f',
                           '#': 'f',
                           't': 'f'
                           }

DIGITS_SYMBOLS_MAPPING = {}

GENERAL_SYMBOLS = "RNBKQabcdefghO-"
DIGIT_SYMBOLS = "12345678"

# Probability to rank conversion
MAX_RANK = 2.5
PROB_TO_RANK = lambda p: math.log10(p) + MAX_RANK if p >= math.pow(10,-MAX_RANK) else 0
RANK_TO_PROB = lambda r: math.pow(10, r - MAX_RANK)

# Penalty for skipping place in rank calculation
SKIP_PLACE_PENALTY = 0.1

# Unfortunatelly digit accurance is not very good so we multiply digit rank by some coefficent (such that digit rank will
# not be greater than 1)
DIGIT_RANK_FACTOR = 0.7

########################################################################################################################
# Classes
########################################################################################################################
class SymbolOcrData:
    """ Class contains ocr info for one symbol"""

    def __init__(self, symbol_ranks, digits_ranks):
        self.symbol_ranks = symbol_ranks
        """ Map symbol-> rank """
        self.digits_ranks = digits_ranks
        """ Map digit -> rank """

    @classmethod
    def probabilityDictToRanks(cls, prob_dict, symbols_mapping, permitted_symbols):
        """ Converts probability dictionary to ranks dictionary

        Parameters
        ----------------
        prob_dict : Dict[char,float]
            probability dictionary
        symbols_mapping : Dict[char,char]
            mapping of symbols which look similar
        permitted_symbols : str
            permitted symbols
        max_threshold_value: float
            if maximal probability is greater than this parameter - don't insert

        Returns
        ----------------
        Dict[char,float]
            Ranks dictionary
        """
        for k, v in symbols_mapping.items():
            if k in prob_dict:
                prob_dict[v] = prob_dict.pop(k) + prob_dict[v] if v in prob_dict else prob_dict.pop(k)

        prob_dict = {k: v for k, v in prob_dict.items() if k in permitted_symbols}

        # Convert to ranks, sort and append
        ranks_list = [(k, PROB_TO_RANK(v)) for k, v in prob_dict.items()]
        ranks_list.sort(key=lambda x: x[1], reverse=True)
        return {k: v for k, v in ranks_list}

    @classmethod
    def from_detected_symbol(self, detected_symbol):
        """ Creates ocr data from symbol

        Parameters
        ----------------
        detected_symbol : DetectedSymbol
            result of OCR engine

        Returns
        ----------------
        Optional[SymbolOcrData]
            None if symbol must be dropped, or new SymbolOcrData otherwise
        """
        symbol_ranks = SymbolOcrData.probabilityDictToRanks(detected_symbol.symbol_prob_dict,
                                                            GENERAL_SYMBOLS_MAPPING,
                                                            GENERAL_SYMBOLS)
        digit_ranks = SymbolOcrData.probabilityDictToRanks(detected_symbol.digit_prob_dict,
                                                           DIGITS_SYMBOLS_MAPPING,
                                                           DIGIT_SYMBOLS)
        if len(symbol_ranks) == 0 and len(digit_ranks) == 0:
            return None
        return SymbolOcrData(symbol_ranks, digit_ranks)

    def __str__(self):
        """ Prints class content to string

        Returns
        ----------------
        str
            class content
        """
        result_array=[]
        for rank_dict, name in [(self.symbol_ranks, "Symbols:"), (self.digits_ranks, ", Digits:")]:
            first = False
            result_array.append(f"{name} {{")
            for k, v in sorted(rank_dict.items(), key=lambda x: x[1], reverse=True):
                result_array.append(f"{', ' if first else ''}['{k}'->{v : .2f}]")
                first = True
            result_array.append("}")
        return "".join(result_array)

    def __repr__(self):
        return str(self)

    def to_html(self):
        """ Prints class content to string in html format

        Returns
        ----------------
        str
            class content in html format
        """

        result_array=[]
        for rank_dict, name in [(self.symbol_ranks, "Symbols"), (self.digits_ranks, "Digits")]:
            result_array.append(f"<h4>{name}</h4><table>")
            i=0
            for k, v in sorted(rank_dict.items(), key=lambda x: x[1], reverse=True):
                if i == 0:
                    result_array.append(f"<tr style=\"background-color:#87CEFA\">")
                    first_index=len(result_array)
                    result_array+=[""]*6
                    result_array.append("</tr><tr>")
                    second_index=len(result_array)
                    result_array+=[""]*6
                    result_array.append("</tr>")
                result_array[first_index+i]=f"<td><b>{k}</b></td>"
                result_array[second_index+i]=f"<td>{v : .2f}</td>"
                i=(i+1)%6
            result_array.append("</table>")
        return "".join(result_array)

    def join(self, other):
        """ Joins other symbol to this

        Parameters
        ----------------
        other : SymbolOcrData
            other ocr data being added
        """
        for rank_dict, other_rank_dict in [(self.symbol_ranks, other.symbol_ranks),
                                           (self.digits_ranks, other.digits_ranks)]:
            for k, v in other_rank_dict.items():
                rank_dict[k] = max(rank_dict[k], v) if k in rank_dict else v

    def rank(self, c):
        """ Returns rank of the given character

        Parameters
        ----------------
        c : char
            character

        Returns
        ----------------
        float
            rank
        """
        if c.isdigit():
            return MAX_RANK*(1-DIGIT_RANK_FACTOR) + DIGIT_RANK_FACTOR*self.digits_ranks[c] if c in self.digits_ranks else 0
        else:
            return self.symbol_ranks[c] if c in self.symbol_ranks else 0

    def normalize(self):
        """ Normalizes ranks to probability equals to 1.0

        """
        for rank_dict in [self.symbol_ranks, self.digits_ranks]:
            if len(rank_dict.items()) == 0:
                continue
            total_prob = sum([RANK_TO_PROB(r) for r in rank_dict.values()])
            for k, v in rank_dict.items():
                rank_dict[k] = v - math.log10(total_prob)


class ImageOcrData:
    """ Class contains list of rank dictionaries
    """

    def __init__(self):
        self.symbols_ocr_data = []

    def __str__(self):
        """ Prints class content to string
        """
        result_array = []
        for i in range(len(self.symbols_ocr_data)):
            result_array.append(f"({i}: {self.symbols_ocr_data[i]})\n")
        return "".join(result_array)

    def __repr__(self):
        return str(self)

    def can_join(self, other):
        """ Returns True if other list can be jointed to this one

        Parameters
        ----------------
        other : RankDictsList
            other list - candidate to join

        Returns
        ----------------
        bool
            True if lists can be joined, False otherwise
        """
        return len(self.symbols_ocr_data) == len(other.symbols_ocr_data)

    def join(self, other):
        """ Joins other list to this, assumed that list can be joined

        Parameters
        ----------------
        other : RankDictsList
            other list - candidate to join
        """
        assert self.can_join(other)
        for i in range(len(self.symbols_ocr_data)):
            self.symbols_ocr_data[i].join(other.symbols_ocr_data[i])

    def add_symbol(self, detected_symbol):
        """ Interface for ocr evaluation, adds next detected with general api symbol to specific picture

        Parameters
        ----------------
        detected_symbol : DetectedSymbol
            detected symbol to be added
        """
        symbol = SymbolOcrData.from_detected_symbol(detected_symbol)
        if symbol is not None:
            self.symbols_ocr_data.append(symbol)

    def str_rank(self, s):
        """ Calculates  string rank.
        Assume string consists of characters c[1]. c[2], ..., c[n] and we stored maps r[1], r[2], r[3], ... r[m] for
        detected symbols 1,...,m. Then

            MAX( Sum (r[l[i]](c[j[i]]) - 0.01*(l[i]-l[i-1] + j[i]-j[i-1]) for all variants where
            1<=j[1]<j[2]...<j[k]<n and 1<=l[1]<l[2]<...<l[k]<m)

        In other words we take some ordered subset of chars in string and check with what subset we have maximal rank.
        For example if we have r[1]={'N'-> 4, 'R' -> 3, 'a' -> 2} and r[2] = {'3' -> 5, 'c' -> 2} then
        move_rank("Nc3")=8.99, move_rank("Rc7")=4.99, moveRank("Rbc3")=7.98 and moveRank("a3")=7.
        Idea is to give to be detected as most symbols as we can, but in case of roughly speaking
        "<Something understandable>f3" returned from OCR we want to give Rf3 and Nf3 more rank than to f3

        Therefore it requires to compare order of 2^(len(move_str)+len(ranks)) numbers but standard notation move size
        is maximum 4 and ocr result will be of same order so it's seems to be ok.

        Parameters
        ----------------
        s : str
            string

        Returns
        ----------------
        float
            string rank
        """
        return ImageOcrData.calc_string_rank(s, self.symbols_ocr_data)


    def get_symbols_number(self):
        return len(self.symbols_ocr_data)

    def to_html(self, max_symbols):
        """ Prints class content to string in html format

        Returns
        ----------------
        str
            class content in html format
        """
        result_array = []
        for i in range(len(self.symbols_ocr_data)):
            result_array.append(f"<td>{self.symbols_ocr_data[i].to_html()}</td>")
        for i in range(len(self.symbols_ocr_data), max_symbols):
            result_array.append("<td></td>")
        return "".join(result_array)

    @classmethod
    def calc_string_rank(cls, s, symbols_ranks):
        """ Recursive function calculates rank for given string for given rank dict list
            See description of move_rank interface
        Parameters
        ----------------
        s : str
            string to calculate the rank
        symbols_ranks : List[Dict[char,float]]
            general ranks dictionaries list

        Returns
        ----------------
        float
            string rank
        """
        if len(s) == 0 or len(symbols_ranks) == 0:
            return -len(s) * SKIP_PLACE_PENALTY

        # choose next indexes in str and ranks
        max_rank = 0
        for i in range(len(s)):
            for j in range(len(symbols_ranks)):
                r_ij = symbols_ranks[j].rank(s[i])
                r_next = ImageOcrData.calc_string_rank(s[i + 1:], symbols_ranks[j + 1:])
                r = + r_ij + r_next - i * SKIP_PLACE_PENALTY
                if r > max_rank:
                    max_rank = r
        return max_rank

    def normalize(self):
        """ After all the values are unified we want to normalize values to probability equals to 1
        """
        for symbol_ocr in self.symbols_ocr_data:
            symbol_ocr.normalize()


class MoveOcrData:
    """ Class which contains ocr data for single move
    """

    def __init__(self):
        self.images_ocr_list = []
        """ List of different unjoinable RankDictList for general (non digits) ocr """
        self.moves_cache = {}
        """ Cache of already calculated ranks"""
        self.is_normalized = False
        """ Flag if list is normailized"""

        self.maximal_rank=0.0
        """ maximal rank from move ranks till  now"""

    def __str__(self):
        """ Prints class content to string
        """
        result_array = []
        i = 0
        for image_ocr in self.images_ocr_list:
            result_array.append(f"----------------- Element {i} ------------------------\n")
            result_array.append(str(image_ocr))
            i += 1
        return "".join(result_array)

    def __repr__(self):
        return str(self)

    def to_html_str(self):
        """ Prints class content to html string as a table
         Returns
        ----------------
        str
            html string
       """
        max_symbols = max([x.get_symbols_number() for x in self.images_ocr_list])

        result_array = ["<table><tr><td>Element</td>"]
        for i in range(max_symbols):
            result_array.append(f"<td>Symbol {i + 1}</td>")
        result_array.append("</tr>")

        i = 0
        for image_ocr in self.images_ocr_list:
            result_array.append(f"<tr><td>{i + 1}</td>")
            result_array.append(image_ocr.to_html(max_symbols))
            result_array.append("</tr>")
            i += 1
        result_array.append("</table>")
        return "".join(result_array)

    def add_image_data(self, new_image_ocr):
        """ Adds image ocr data to this move

        Parameters
        ----------------
        new_image_ocr : ImageOcrData
            ocr data for image
        """
        assert not self.is_normalized
        joined = False
        # for image_ocr in self.images_ocr_list:
        #     if image_ocr.can_join(new_image_ocr):
        #         image_ocr.join(new_image_ocr)
        #         joined = True
        if not joined:
            self.images_ocr_list.append(new_image_ocr)

    def move_rank(self, move_str):
        """ Interface for game building, calculates move string rank.

        Parameters
        ----------------
        move_str : str
            move string in standard notation

        Returns
        ----------------
        float
            move rank
        """
        if move_str in self.moves_cache:
            return self.moves_cache[move_str]
        rank = max([x.str_rank(move_str) for x in self.images_ocr_list])
        self.moves_cache[move_str] = rank
        self.maximal_rank=max(rank, self.maximal_rank)
        return rank

    def max_rank(self):
        """ Interface for game building, calculates move string rank.

        Returns
        ----------------
        float
            max rank
        """
        return self.maximal_rank

    def normalize(self):
        """ After all the values are unified we want to normalize values to probability equals to 1
        """
        for image_ocr in self.images_ocr_list:
            image_ocr.normalize()
        self.is_normalized = True


class MoveOcrDataList :
    """ Class which contains ocr data for single move but from different engines
        Has pretty straighforward getters/setters
    """
    def __init__(self):
        self.move_ocr_list=[]
        """ list of ocr datas"""

    def add_move_ocr(self,m):
        self.move_ocr_list.append(m)

    def normalize(self):
        for m in self.move_ocr_list:
            m.normalize()

    def calc_ranks(self, move_str):
        return [m.move_rank(move_str) for m in self.move_ocr_list]

    def get_distance(self, move_str):
        ranks=self.calc_ranks(move_str)
        return min([self.move_ocr_list[i].max_rank() - ranks[i] for i in range(len(self.move_ocr_list))])



