function next_prev(is_next)
{
    var active_elem=document.getElementsByClassName("step_active")[0];
    var active = parseInt(active_elem.id.substring(4,active_elem.id.length));
    var fs_elem= document.getElementById("fs"+active.toString())


    var step_elements=document.getElementsByClassName("step");
    var max = 0;
    for (var step_element of step_elements)
    {
        index=parseInt(step_element.id.substring(4,step_element.id.length));
        if (index > max)
        {
            max=index;
        }
    }
    if(active > max) max = active;

    if(is_next)
    {
        if(active == max) return;
        active++;
    }
    else
    {
        if(active == 1) return;
        active--;
    }

    // run python staff
    switch(active)
    {
    case 1:
        eel.wizard_step1();
        break;

    case 2:
        document.getElementById("build_cells_progress_block").style.display="block"
        document.getElementById("build_cells_progress").value=0
        document.getElementById("cellsTable").innerHTML=""
        var img=document.getElementById("gameImage").src
        if((img == "") || (document.getElementById("gameImage").style.display=="none"))
            return
        eel.wizard_step2(img);
        break;

    case 3:
        document.getElementById("finalMovesTable").innerHTML=""
        eel.wizard_step3();
        break;
    case 4:
        document.getElementById("game_tree_container").innerHTML=""
        eel.wizard_step4();
        break;


    default:
        break;
    }

    var next_elem = document.getElementById("step"+active.toString());
    next_elem.classList.add("step_active");
    next_elem.classList.remove("step");
    active_elem.classList.add("step");
    active_elem.classList.remove("step_active");
    if(active == 1)
    {
        document.getElementById("prev").style.display = "none";
    }
    else
    {
        document.getElementById("prev").style.display = "block";
    }
    if(active == max)
    {
        document.getElementById("next").style.display = "none";
    }
    else
    {
        document.getElementById("next").style.display = "block";
    }
    fs_elem.style.display="none"
    document.getElementById("fs"+active.toString()).style.display="block"
}

function onFileNameChange()
{
    const preview = document.getElementById('gameImage');


    const file = document.getElementById('filename').files[0];
    const reader = new FileReader();


    reader.addEventListener("load", function () { preview.src = reader.result; preview.style.display="block"}, false);

    if (file) {
        reader.readAsDataURL(file);
    }
}
function onGameImageLoadFailure()
{
    console.log("onGameImageLoadFailure")
    document.getElementById('gameImage').src="";
    document.getElementById('gameImage').style.display="none";

}

document.getElementById("next").addEventListener("click", ()=>{next_prev(true);}, false);
document.getElementById("prev").addEventListener("click", ()=>{next_prev(false);}, false);
document.getElementById("filename").addEventListener("change", ()=> {onFileNameChange();}, false);


function onColumnRemove(event)
{
    eel.remove_column(event.srcElement.parentElement.cellIndex-1);
}
function onRowRemove(event)
{
    eel.remove_row(event.srcElement.parentElement.parentElement.rowIndex-1);
}

eel.expose(buildCellsTable)
function buildCellsTable(rows, columns)
{
    try
    {
        document.getElementById("build_cells_progress_block").style.display="none"
        var table_elem=document.getElementById("cellsTable");
        // Clear table
        table_elem.innerHTML=""

        var i,j;

        // Add remove column buttons
        var buttons_row=table_elem.insertRow();
        buttons_row.insertCell();
        for (i=0;i<columns; i++)
        {
            var button = document.createElement("input");
            button.type="button";
            button.value="Remove"
            buttons_row.insertCell().appendChild(button);

            button.addEventListener("click", ( event ) => {onColumnRemove(event);}, true)
        }

        // add cells
        for (i=0; i<rows; i++)
        {
            // add remove row button
            var row=table_elem.insertRow();
            var button = document.createElement("input");
            button.type="button";
            button.value="Remove"
            row.insertCell().appendChild(button);
            button.addEventListener("click", ( event ) => {onRowRemove(event);}, true)


            for (j=0; j<columns; j++)
                cell=row.insertCell();
        }
        // put images
        for (i=0; i<rows; i++)
            for(j=0; j<columns; j++)
                eel.put_cell_img(i,j,"cellsTable");
    }


    catch (error)
    {
      console.error(error);
    }
}

eel.expose(buildMovesTable)
function buildMovesTable(rows, columns)
{
    console.log(rows);
    console.log(columns);
    try
    {
        var table_elem=document.getElementById("finalMovesTable");
        // Clear table
        table_elem.innerHTML=""

        var i,j;

        // Add headers cells
        row=table_elem.insertRow();
        row_html="";
        for (i=0;i<columns/2; i++)
        {
            row_html+="<th>#</th><th>White</th><th>Black</th>";
        }
        row.innerHTML=row_html;

        // Add cells
        for (i=0; i<rows; i++)
        {
            // add remove row button
            var row=table_elem.insertRow();
            for (j=0; j<columns/2; j++)
            {
                var move_cell=row.insertCell();
                move_cell.innerHTML=(j*rows+i+1).toString();
                var white_cell=row.insertCell();
                var black_cell=row.insertCell();
            }
        }
        // put images
        for (i=0; i<rows; i++)
        {
            for(j=0; j<columns; j++)
            {
                eel.put_cell_img(i,j,"finalMovesTable");
            }
        }
    }

    catch (error)
    {
      console.error(error);
    }
}

eel.expose(putCellImage)
function putCellImage(i,j,table_name,img)
{
    try
    {
        switch(table_name)
        {
        case "cellsTable":
            row=i+1;
            column=j+1;
            break;
        case "finalMovesTable":
            row=i+1;
            column=j+Math.floor(j/2)+1;
            break;
        default:
            alert("Unknow table name: "+ table_name);
            return;
        }
        document.getElementById(table_name).rows[row].cells[column].innerHTML="<img src=\""+img + "\"/>";
    }
    catch (error)
    {
      console.error(error);
    }

}

eel.expose(updateBuildCellsProgress)
function updateBuildCellsProgress(progressValue)
{
    document.getElementById("build_cells_progress").value=progressValue
}

eel.expose(showDebugGameTree)
function showDebugGameTree(html_str)
{
    document.getElementById("game_tree_container").innerHTML=html_str
    var toggler = document.getElementById("game_tree_container").getElementsByClassName("caret");
    var i;

    for (i = 0; i < toggler.length; i++) {
      toggler[i].addEventListener("click", function() {
        this.parentElement.querySelector(".nested").classList.toggle("active");
        this.classList.toggle("caret-down");
      });
    }
}
