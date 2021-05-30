from game_image import GameImage, CellExtraEval
from tess_api import ImagesSupplier, build_moves_ocr_data, TRAIN_LIST
from move_ocr_data import MoveOcrData, MoveOcrDataList
import chess
import chess.pgn
import treelib
import cv2
import base64
import time
import numpy as np
import threading
import queue
import cProfile

STOP_AT_MOVE = 12
START_THRESHOLD = 1.5
MAX_VARIATIONS_DEPTH = 15
DROP_PARAM_START = 0
DROP_PARAM_STEP = 0.05
DROP_PARAM_STEPS_NUMBER = 60
MAX_PGN_LEAFS_NUMBER = 600000
THREADS_NUMBER = 5

def get_drop_threshold(param, depth):
    """ Calculates threshold according to parameter and depth

    Parameters
    ----------------
    param : float
        param value
    depth: int
        depth of the pgn tree of the node
    Returns
    ----------------
    float
        threshold
    """
    if depth < MAX_VARIATIONS_DEPTH:
        return START_THRESHOLD * np.power(((MAX_VARIATIONS_DEPTH - depth) / MAX_VARIATIONS_DEPTH), param)
    else:
        return 0.0001


def get_param_index_for_threshold(threshold, depth):
    """ Calculates minimal N that get_drop_threshold(DROP_PARAM_START+N*DROP_PARAM_STEP) > threshold

    Parameters
    ----------------
    threshold : float
        param value
    depth: int
        depth of the pgn tree of the node
    Returns
    ----------------
    int
        threshold
    """
    if threshold < 0.0001:
        return DROP_PARAM_STEPS_NUMBER

    if depth >= MAX_VARIATIONS_DEPTH or threshold > START_THRESHOLD - 0.0001:
        return 0

    if depth == 0:
        return DROP_PARAM_STEPS_NUMBER

    param = np.log(threshold / START_THRESHOLD) / np.log((MAX_VARIATIONS_DEPTH - depth) / MAX_VARIATIONS_DEPTH)
    return int((param - DROP_PARAM_START) / DROP_PARAM_STEP) + 1




########################################################################################################################
# Game ocr class
########################################################################################################################
class GameOcr:
    class ThreadPool:
        def __init__(self, number):
            self.queue = queue.Queue()
            self.threads = []
            for i in range(number):
                thread = threading.Thread(target=self.func, args=())
                self.threads.append(thread)
                thread.start()

        def func(self):
            print(f"Thread {threading.get_ident()} started")
            while (True):
                operation = self.queue.get()
                if operation[0] is None:
                    self.queue.task_done()
                    break
                operation[0](operation[1])
                self.queue.task_done()
            print(f"Thread {threading.get_ident()} stopped")

        def map(self,f,arg_list):
            for arg in arg_list:
                self.queue.put((f,arg))

        def wait_for_finished(self):
            self.queue.join()

        def close(self):
            self.map(None,[None]*len(self.threads))
            for thread in self.threads:
                thread.join()

    """ Class responsible for game evaluation
    """
    class GameImageSupplier(ImagesSupplier):
        """ Helper class supplies images to engine
        """

        def __init__(self, game_image):
            """ Constructor
            Parameters
            ----------------
            game_image : GameImage
                Game image class. We assumed that table is built and extra rows/columns removed
            """
            self.game_image = game_image
            """ Game image which cells will be supplied"""

            self.index = 0
            """ Image index - we assume that it is (column/2)*column_size+row+column%2 """

            self.eval_modes_list = list(map(int, CellExtraEval))

        def get_next_image(self):
            """ Returns next image, or None if all images were evaluated

            Returns
            ----------------
            Optional[int,np.array]]
                tuple (index, image), where index is semi-move index and image - image in cv2 format
            """
            column_size = self.game_image.get_table_rows_number()
            column_index = 2 * int(self.index / (2 * column_size)) + self.index % 2
            row_index = int(self.index / 2) % column_size
            if column_index >= self.game_image.get_table_columns_number():
                return None

            if self.index == STOP_AT_MOVE:
                return None

            eval_mode = self.eval_modes_list.pop(0)
            ret_value = (
                (self.index, eval_mode), self.game_image.get_colored_cell((row_index, column_index), eval_mode))
            if len(self.eval_modes_list) == 0:
                self.index += 1
                self.eval_modes_list = list(map(int, CellExtraEval))
            return ret_value

    def build_moves_data(self):
        """ Builds moves ocr data
        """
        rows = self.game_image.get_table_rows_number()
        columns = self.game_image.get_table_columns_number()
        moves_number = min(STOP_AT_MOVE, rows * columns)
        self.moves_data = []
        for i in range(moves_number):
            self.moves_data.append(MoveOcrDataList())

        """ list of ocr datas for semi_moves"""
        for train_id in range(len(TRAIN_LIST)):
            print(f"Running detection for {TRAIN_LIST[train_id]}")

            training_data = build_moves_ocr_data(GameOcr.GameImageSupplier(self.game_image), train_id=train_id)
            items = [x for x in training_data.items()]
            items.sort(key=lambda x: x[0])

            i = 0
            index = 0
            while i < len(items):
                assert items[i][0][0] == index
                m = MoveOcrData()
                j = i
                while j < len(items) and items[j][0][0] == index:
                    m.add_image_data(items[j][1])
                    j += 1
                self.moves_data[index].add_move_ocr(m)
                i = j
                index += 1

        for m in self.moves_data:
            m.normalize()

    def __init__(self, game_image):
        """ Constructor

        Parameters
        ----------------
        game_image : GameImage
            Game image class. We assumed that table is built and extra rows/columns removed
        """
        self.game_image = game_image
        """ Game image """

        self.moves_data = []
        """ List, index is half move, value MoveOcrDataList """

        self.chess_game = chess.pgn.Game()
        """ Chess game with all the variations """

        self.moves_str_cache = {}
        """ Nodes->move string cache dictionary"""

        self.build_moves_data()

    def get_move_san(self, node, board):
        """ Returns san of node's move. If found in cache returns from cache, otherwise calculates

        Parameters
        ----------------
        node : chess.pgn.GameNode
            Node
        board: chess.Board
            board corresponds to position without node move

        Returns
        ----------------
        str
            Move string in san format
        """

        if node in self.moves_str_cache:
            return self.moves_str_cache[node]
        else:
            move_str = board.san(node.move)
            self.moves_str_cache[node] = move_str
            return move_str

    def remove_nodes(self, node, board, drop_param, depth):
        """ Removes nodes according to threshold function

        Parameters
        ----------------
        node : chess.pgn.Node
            node for which tree we are going to build next level
        board : chess.pgn.Board
            board corresponds to node, needed cause node.board() is relatively slow
        drop_param: float
            drop_param for threshold function
        depth: int
            depth of the pgn tree of the node

        Returns
        ----------------
        Tuple(int,int)
            (number of nodes in tree, number of legal moves from leafs)
        """
        if depth < 0:
            return (1, board.legal_moves.count())

        min_distance = min([float(v.comment) for v in node.variations])
        node.variations = [v for v in node.variations
                           if float(v.comment) < min_distance + get_drop_threshold(drop_param, depth)]
        nodes_number = 0
        leafs_legal_moves_number = 0

        # Go over variations calculate ranks
        for v in node.variations:
            board.push(v.move)
            v_nodes_number, v_legal_moves = self.remove_nodes(v, board, drop_param, depth - 1)
            board.pop()
            nodes_number += v_nodes_number
            leafs_legal_moves_number += v_legal_moves

        return (nodes_number + 1, leafs_legal_moves_number)

    def update_distances(self, node, board, half_move, depth):
        """ Updates distances after level is build and max_rank information is calculated for all half-moves

        Parameters
        ----------------
        node : chess.pgn.Node
            node for which tree we are going to build next level
        board : chess.pgn.Board
            board corresponds to node, needed cause node.board() is relatively slow
        half_move: int
            half move of the node
        depth: int
            depth of the pgn tree of the node

        Returns
        ----------------
        Tuple(float,np.array[int])
            Tuple(Minimal distance in tree, list of parameters to legal moves count)
        """
        if depth < 0:
            legal_moves = board.legal_moves.count()
            return (0, np.full((DROP_PARAM_STEPS_NUMBER), legal_moves, dtype=int))


        legal_moves = np.full((DROP_PARAM_STEPS_NUMBER), 0, dtype=int)
        v_legal_moves = {}
        if depth == 0:
            # For legal moves we choose one random and update all according to it to save time (it's estimation)
            if len(node.variations) > 0:
                index = np.random.randint(0,len(node.variations))
                board.push(node.variations[index].move)
                _,rand_legal_moves=  self.update_distances(node.variations[index], board, half_move + 1, depth - 1)
                board.pop()

        min_dist = 10000.0
        for v in node.variations:
            if depth != 0:
                board.push(v.move)
                v_distance, v_legal_moves[v] = self.update_distances(v, board, half_move + 1, depth - 1)
                board.pop()
            else:
                v_distance=0
                v_legal_moves[v]=rand_legal_moves

            # take care of the distance
            v_distance += self.moves_data[half_move].get_distance(self.get_move_san(v, board))
            v.comment = f"{v_distance : .2f}"
            min_dist = min(min_dist, v_distance)

        for v in node.variations:
            # take care of legal moves
            index = get_param_index_for_threshold(float(v.comment) - min_dist, depth)
            legal_moves[0:index - 1] = np.add(legal_moves[0:index - 1], v_legal_moves[v][0:index - 1])

        # remove all those not passing initial maximal threshold
        node.variations = [v for v in node.variations
                           if float(v.comment) < min_dist + get_drop_threshold(DROP_PARAM_START, depth)]

        node.variations.sort(key=lambda v: float(v.comment))
        return (min_dist, legal_moves)

    def add_children_to_leaf(self, leaf):
        board=leaf.board()
        for m in board.legal_moves:
            v=leaf.add_variation(m)
            move_str=self.get_move_san(v,board)
            self.moves_data[board.ply()].calc_ranks(move_str)

        min_dist=10000.0
        for v in leaf.variations:
            v_distance=self.moves_data[board.ply()].get_distance(self.get_move_san(v, board))
            v.comment = f"{v_distance : .2f}"
            min_dist = min(min_dist, v_distance)

        # remove all those not passing initial maximal threshold
        leaf.variations = [v for v in leaf.variations
                           if float(v.comment) < min_dist + get_drop_threshold(DROP_PARAM_START, 0)]
        leaf.variations.sort(key=lambda v: float(v.comment))


    def build_level_from_leafs(self, leafs_list):
        self.threadPool.map(self.add_children_to_leaf, leafs_list)
        self.threadPool.wait_for_finished()

    def get_pgn_leafs_list(self, node, board, depth):
        if depth == 0:
            return [node]
        result = []
        for v in node.variations:
            board.push(v.move)
            result += self.get_pgn_leafs_list(v,board, depth-1)
            board.pop()
        return result

    def build_pgn_level(self, node, board, half_move, depth):
        """ Builds next level for the pgn tree with specific depth

        Parameters
        ----------------
        node : chess.pgn.Node
            node for which tree we are going to build next level
        board : chess.pgn.Board
            board corresponds to node, needed cause node.board() is relatively slow
        half_move: int
            half move from the beginning of the game startin from 0.
        depth: int
            depth of the pgn tree of the node

        Returns
        ----------------
        float,
            rank of the tree
        """

        if half_move == len(self.moves_data):
            return 0

        if depth < 0:
            return 0

        if depth == 0:
            # If it is level that we need to build, just add all moves, next cycle will remove uneeded
            for m in board.legal_moves:
                node.add_variation(m)

        # Go over variations calculate ranks
        for v in node.variations:
            self.moves_data[half_move].calc_ranks(self.get_move_san(v, board))

        min_dist = 10000.0
        for v in node.variations:
            board.push(v.move)
            v_distance = self.build_pgn_level(v, board, half_move + 1, depth - 1)
            board.pop()
            v_distance += self.moves_data[half_move].get_distance(self.get_move_san(v, board))
            v.comment = f"{v_distance : .2f}"
            min_dist = min(min_dist, v_distance)

        # remove all those not passing initial maximal threshold
        node.variations = [v for v in node.variations
                           if float(v.comment) < min_dist + get_drop_threshold(DROP_PARAM_START, depth)]
        node.variations.sort(key=lambda v: float(v.comment))
        return min_dist

    def calculate_drop_param(self, leaf_counts):
        """ Calculates

        Parameters
        ----------------
        leaf_counts : np.array
            index -> number of leafs in next tree would remain if threshold param is DROP_PARAM_STEP*i+DROP_PARAM_START
        """

        index = DROP_PARAM_STEPS_NUMBER - np.searchsorted(np.flip(leaf_counts), MAX_PGN_LEAFS_NUMBER)
        print(f"\tIndex={index}")
        return DROP_PARAM_STEP * index + DROP_PARAM_START

    def evaluate_pgn_tree_level(self, level, is_last):
        """ Performs operations need for building pgn tree level - builds, updates distances and removes

        Parameters
        ----------------
        level : int
            level index
        is_last: bool
            flag whether this level is last
        """

        operations_list = \
            [("Build leafs",
              False,
              lambda self, ret, level, is_last: self.get_pgn_leafs_list(self.chess_game,
                                                                        self.chess_game.board(),
                                                                        level)),
             ("Build level from list",
              False,
              lambda self, ret, level, is_last: self.build_level_from_leafs(ret)),

              # ("Build tree level", lambda self, ret, level, is_last: self.build_pgn_level(self.chess_game,
              #                                                                              self.chess_game.board(),
              #                                                                              0,
              #                                                                              level)),
              ("Update distances",
               True,
               lambda self, ret, level, is_last: self.update_distances(self.chess_game,
                                                                       self.chess_game.board(),
                                                                       0,
                                                                       level)),
              ("Remove nodes",
               True,
               lambda self, ret, level, is_last: self.remove_nodes(self.chess_game,
                                                                   self.chess_game.board(),
                                                                   self.calculate_drop_param(ret[1]),
                                                                   level) if not is_last else None)]
        print(f"Starting evaluating pgn tree level {level}")
        total_time_before = time.time()
        ret = None
        for operation in operations_list:
            print(f"\tStarting operation \"{operation[0]}\"")
            time_before = time.time()
            # if operation[0] == "Update distances":
            #     profile=cProfile.Profile()
            #     profile.enable()
            #     ret = operation[2](self, ret, level, is_last)
            #     profile.disable()
            #     profile.print_stats("cumtime")
            # else:
            ret = operation[2](self, ret, level, is_last)

            print(f"\tOperation \"{operation[0]}\"finished. "
                  f"Elapsed time: {time.time() - time_before}",end="")
            if operation[1]:
                print(f", return value: {ret}\n",end="")
            else:
                print("\n",end="")


        print(f"Finished evaluating pgn tree level {level}.\n"
              f"Elapsed time: {time.time() - total_time_before}.\n"
              f"Estimated time for next level: {0 if ret == None else ret[1] * 4.93 / 100000}")

    def build_pgn(self):
        """ Builds pgn
        """
        self.chess_game = chess.pgn.Game()
        self.threadPool = GameOcr.ThreadPool(THREADS_NUMBER)
        for i in range(min(STOP_AT_MOVE, len(self.moves_data))):
            self.evaluate_pgn_tree_level(i, i + 1 == min(STOP_AT_MOVE, len(self.moves_data)))
        self.threadPool.close()
        self.threadPool = None
        print("Finished building pgn")

    def debug_txt_node_tree(self, node, board, tree):
        nodes_number = 0
        index = 0

        for v in node.variations:
            v_str = f"{index} - {self.get_move_san(v, board)} : {float(v.comment):.2f}"
            tree.create_node(v_str, id(v), parent=id(node))
            board.push(v.move)
            nodes_number += self.debug_txt_node_tree(v, board, tree)
            board.pop()
            index = index + 1
        return nodes_number + 1

    def debug_txt_tree_show(self):
        tree = treelib.Tree()
        tree.create_node("Origin", id(self.chess_game))
        n = self.debug_txt_node_tree(self.chess_game, chess.pgn.Game().board(), tree)
        tree.show()

    def debug_html_node_tree(self, node, board, half_move, move_str):
        if half_move > 0:
            move_ranks = self.moves_data[half_move - 1].calc_ranks(move_str)
            rank_str = f"{float(node.comment) :.2f} ({move_ranks})"
        else:
            rank_str = ""

        nodeStr = f"{move_str} :  {rank_str}"
        self.html_tree_array.append(f"<li><span class=\"caret\">{nodeStr} </span><ul class=\"nested\">")
        nodes_number = 0
        index = 0
        for v in node.variations:
            v_str = self.get_move_san(v, board)
            board.push(v.move)
            nodes_number += self.debug_html_node_tree(v, board, half_move + 1, v_str)
            board.pop()
            index = index + 1
        self.html_tree_array.append("</ul></li>")
        return nodes_number + 1

    def debug_html_tree_get(self):
        self.html_tree_array = [""]
        ret = self.debug_html_node_tree(self.chess_game, chess.pgn.Game().board(), 0, "Game")
        print(f"Number of nodes={ret}")
        self.html_tree_array[0] = f"<h4> Number of nodes={ret}</h4>"

        # for i in range(len(self.moves_data)):
        #     column_index = 2 * int(i / (2 * self.game_image.get_table_rows_number())) + i % 2
        #     row_index = int(i / 2) % self.game_image.get_table_rows_number()
        #     image = self.game_image.get_colored_cell((row_index, column_index))
        #     _, im_arr = cv2.imencode(".jpg", image)  # im_arr: image in Numpy one-dim array format.
        #     im_b64 = base64.b64encode(im_arr)
        #     self.html_tree_array.append(
        #         f"<img src=\"data:image/jpg;charset=utf-8;;base64,{im_b64.decode('utf-8')} \"></img>")
        #     self.html_tree_array.append(self.moves_data[i].to_html_str())
        return "".join(self.html_tree_array)

    def debug_calc_correct_moves_diffs(self, board, half_move, correct_moves):
        if half_move == len(self.moves_data):
            return []

        for move in board.legal_moves:
            self.moves_data[half_move].calc_ranks(board.san(move))

        result = None
        correct_dist = None
        for move in board.legal_moves:
            move_str = board.san(move)

            if move_str == correct_moves[0][0]:
                correct_dist = self.moves_data[half_move].get_distance(move_str)
                board.push(move)
                result = self.debug_calc_correct_moves_diffs(board, half_move + 1, correct_moves[1:])
                board.pop()

        result.insert(0, (correct_moves[0][1], correct_dist))
        return result


def test():
    game_image = GameImage.from_filename("games/small.jpg")
    game_image.build_table_cells(None)
    game_image.remove_row(0)
    game_image.remove_row(12)
    game_image.remove_column(2)
    print("Image is loaded")

    correct_moves = [('e4', 'e4'), ('c5', 'c5'), ('Nf3', 'Nf3'), ('Nc6', 'Nc6'), ('d4', 'd4'), ('cxd4', 'cxd4'),
                     ('Nxd4', 'Nxd4'), ('Nf6', 'Nf6'), ('Nc3', 'Nc3'), ('e5', 'e5'), ('Ndb5', 'Nb5'), ('d6', 'd6'),
                     ('Bg5', 'Bg5'), ('a6', 'a6'), ('Na3', 'Na3'), ('Be6', 'Be6'), ('Nc4', 'Nc4'), ('Rb8', 'Rb8'),
                     ('Nd5', 'Nd5'), ('Bxd5', 'Bxd5'), ('exd5', 'exd5'), ('Ne7', 'Ne7'), ('Bxf6', 'Bxf6'),
                     ('gxf6', 'gxf6')]
    correct_diffs_matrix = [(x[0], []) for x in correct_moves]
    game_ocr = GameOcr(game_image=game_image)
    # for i in range(len(game_ocr.moves_data)):
    #     print(
    #         f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! {i} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #     print(f"{game_ocr.moves_data[i]}\n")
    #    print("BUILDING PGN ")
    #    cProfile.runctx("game_ocr.build_pgn()", {"game_ocr" : game_ocr}, {})
    #    print("\n\nBUILDING HTML TREE")
    #    cProfile.runctx("game_ocr.debug_html_tree_get()", {"game_ocr" : game_ocr}, {})
    game_ocr.build_pgn()
    game_ocr.debug_html_tree_get()
    # game_ocr.debug_tree_show()
    # print(game_ocr.chess_game)
    # print(len(correct_moves))
    # for i in range(len(correct_moves)):
    #     print(f"{game_ocr.moves_data[i].move_rank(correct_moves[i]) : .2f}")

    correct_diffs = game_ocr.debug_calc_correct_moves_diffs(chess.pgn.Game().board(), 0, correct_moves)
    for i in range(len(correct_diffs)):
        correct_diffs_matrix[i][1].append(correct_diffs[i][1])

    for c in correct_diffs_matrix:
        print(f"Move = {c[0].ljust(4)} diffs from maximum     ", end="")
        for v in c[1]:
            print(f"{v : .2f}, ", end="")
        print("\n", end="")

#test()