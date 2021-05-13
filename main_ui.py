import eel
from game_image import GameImage
import cv2
import base64

@eel.expose
def wizard_step2(uri) :
    global game
    game = None
    game=GameImage.from_uri(uri)
    game.build_table_cells()
    eel.buildCellsTable(game.get_table_rows_number(),game.get_table_columns_number())
    return

@eel.expose
def wizard_step3() :
    global game
    eel.buildMovesTable(game.get_table_rows_number(),game.get_table_columns_number())
    return

@eel.expose
def put_cell_img(i,j, table_name) :
    global game
    image = game.get_colored_cell((i,j))
    _, im_arr = cv2.imencode(".jpg", image)  # im_arr: image in Numpy one-dim array format.
    im_b64 = base64.b64encode(im_arr)
    eel.putCellImage(i,j,table_name,f"data:image/jpg;charset=utf-8;;base64,{im_b64.decode('utf-8')}")

@eel.expose
def remove_row(index) :
    global game;
    game.remove_row(index)
    eel.buildCellsTable(game.get_table_rows_number(),game.get_table_columns_number())

@eel.expose
def remove_column(index) :
    global game;
    game.remove_column(index)
    eel.buildCellsTable(game.get_table_rows_number(),game.get_table_columns_number())


def close_callback(route, websockets):
    if not websockets:
        exit()

eel.init('web')
eel.start("index.html")#,cmdline_args=['--incognito'])
