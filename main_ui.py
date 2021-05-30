import eel
from game_image import GameImage
from game_ocr import GameOcr
import cv2
import base64

@eel.expose
def wizard_step1() :
    global game_image
    game_image=None

@eel.expose
def wizard_step2(uri) :
    global game_image
    if game_image == None:
        game_image=GameImage.from_uri(uri)
        game_image.build_table_cells(lambda x: eel.updateBuildCellsProgress(int(x * 100)))
    else:
        game_image.restore_to_initial()
        eel.updateBuildCellsProgress(100)
    eel.buildCellsTable(game_image.get_table_rows_number(), game_image.get_table_columns_number())
    return

@eel.expose
def wizard_step3() :
    global game_image
    eel.buildMovesTable(game_image.get_table_rows_number(), game_image.get_table_columns_number())
    return

@eel.expose
def wizard_step4() :
    global game_image
    global game_ocr
    game_ocr=GameOcr(game_image)
    game_ocr.build_pgn()
    html=game_ocr.debug_html_tree_get()
    eel.showDebugGameTree(html)
    return

@eel.expose
def put_cell_img(i,j, table_name) :
    global game_image
    image = game_image.get_colored_cell((i, j))
    _, im_arr = cv2.imencode(".jpg", image)  # im_arr: image in Numpy one-dim array format.
    im_b64 = base64.b64encode(im_arr)
    eel.putCellImage(i,j,table_name,f"data:image/jpg;charset=utf-8;;base64,{im_b64.decode('utf-8')}")

@eel.expose
def remove_row(index) :
    global game_image;
    game_image.remove_row(index)
    eel.buildCellsTable(game_image.get_table_rows_number(), game_image.get_table_columns_number())

@eel.expose
def remove_column(index) :
    global game_image;
    game_image.remove_column(index)
    eel.buildCellsTable(game_image.get_table_rows_number(), game_image.get_table_columns_number())


def close_callback(route, websockets):
    if not websockets:
        exit()


game_image=None
game_ocr=None
eel.init('web')
eel.start("index.html")#,cmdline_args=['--incognito'])
