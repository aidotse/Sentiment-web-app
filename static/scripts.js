var result_data
filename = ""
var table_data
var classification_breakpoint = 0

var available_models = {"NER":["fear_target" , 'violence_target', "swedish_ner"],
                        "sentiment":["fear_sentiment", "violence_sentiment"]}

var mdl_srt = {
    'violence_sentiment': ['unsorted', 'sorted','max'],
    'fear_sentiment': ['unsorted', 'sorted','max'],
    'fear_target': ['ner-prediction', 'ner-classification'],
    'violence_target': ['ner-prediction', 'ner-classification'],
    'swedish_ner':['ner-prediction', 'ner-classification']
}


function clearBox(elementID){
                            document.getElementById(elementID).innerHTML = "";
                            };
// initiate the result table 
$(document).ready(function(){
         $('#res_table').bootstrapTable({
        search: true,
        columns: [{
                    field: 'message',
                    title: 'Message',
                    sortable: true
                }, {
                    field: 'pred',
                    title: 'Prediction',
                    sortable: true
                },{
                    field: 'clss',
                    title: 'Classification',
                    sortable: true
                }, ],
                
                data: table_data
            })
        })

// init popovers 
$(document).ready(function(){
    $('[data-toggle="popover"]').popover();
})

function convertToCSV(objArray) {
                        var array = typeof objArray != 'object' ? JSON.parse(objArray) : objArray;
                        var str = '';

                        for (var i = 0; i < array.length; i++) {
                            var line = '';
                            for (var index in array[i]) {
                                if (line != '') line += ','

                                line += array[i][index];
                                }

                                str += line + '\r\n';
                                }

                                return str;
                            }

function ajaxOnSubmit(){clearBox('info_box');
                        $(".info_box").append(`Processing the data`);}

function select_classification_breakpoint(mod_select){
    if (mod_select == 'violence_sentiment'){
        classification_breakpoint = 0.575}
    else if (mod_select == 'fear_sentiment'){
        classification_breakpoint = 0.5}
    return classification_breakpoint}

function ajaxOnError(ajaxResponse, errorStr){
                        var msg = '';
                        clearBox('info_box');
                        $("#loader").hide();
                        if (ajaxResponse.status === 500) {
                                msg = 'Something went wrong, check you inputs and try again';
                        };
                        $(".info_box").append(msg);
                        }
function ajaxCallback(data){clearBox('info_box');
                            clearBox('msg');
                            result_data = data;
                            classification_breakpoint = select_classification_breakpoint($("#model-select").val())
                            if (data.message == "no_data_uploaded_or_in_text_area_"){
                                $(".info_box").append(`No data available`);
                                $("#res_table_div").hide()
                                }
                            else if(available_models["sentiment"].includes($("#model-select").val()) ){
                                // If predictionresults are present, load them in the table, allways 
                                var table_data = [];
                                var classification_all = [];
                                    for (i = 0; i < data.message.length; i++){
                                        // do classification for each prediction
                                        if(data.pred[i] < classification_breakpoint){
                                            // if pred < breakpoint 
                                            clss = 0
                                        }
                                        else{
                                            // else classify as positive
                                            clss = 1
                                        }
                                        new_line = {'message':data.message[i], 'pred':(data.pred[i]).toFixed(2), 'clss':clss};
                                        classification_all.push(clss);
                                        table_data[i]=new_line;
                                        };

                                    result_data.classification = classification_all;
                                    $('#res_table').bootstrapTable('load', table_data);
                                    $("#res_table_div").show()
                                    }
                            else if(available_models["NER"].includes($("#model-select").val()) ){
                                $("#res_table_div").hide();
                                $('.info_box').append(`NER models can not yet be visualized, but can be exported as a JSON`);
                                console.log(data)
                            }
                            
                            else {
                                console.log(data)}
                                }
                                
                                

function exportJson(el) {   if (available_models["sentiment"].includes($("#model-select").val()) ) {
                                var data = "text/json;charset=utf-8," + encodeURIComponent(JSON.stringify({"message":result_data.message,"prediction":result_data.pred}));
                            }
                            else if(available_models["NER"].includes($("#model-select").val()) ){
                                var data = "text/json;charset=utf-8," + encodeURIComponent(JSON.stringify({"message":result_data.message,"prediction":result_data.pred, "NER-labels":result_data.NER_labels}));
                            }
                            
                            el.setAttribute("href", "data:"+data);
                            el.setAttribute("download", "results.json");}

function exportCSV(el) {    var csv_string = "";
                            for (i = 0; i < result_data.message.length; i++){
                                tt = result_data.message[i].replace(/,/g,';') // replace commas with semi-commas (, -> ;) to avoid erros in the csv
                                csv_string += `${tt},${result_data.pred[i]}, ${result_data.classification[i]}\r\n`}
                            
                            var csv = csv_string

                            var blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
                            if (navigator.msSaveBlob) { // IE 10+
                                navigator.msSaveBlob(blob, 'result.csv');
                            } else {
                                var link = document.createElement("a");
                                if (link.download !== undefined) { // feature detection
                                    // Browsers that support HTML5 download attribute
                                    var url = URL.createObjectURL(blob);
                                    link.setAttribute("href", url);
                                    link.setAttribute("download", 'result.csv');
                                    link.style.visibility = 'hidden';
                                    document.body.appendChild(link);
                                    link.click();
                                    document.body.removeChild(link);
                                        }
                                    }
                                }



$(document).ready(function() {
    // User Submit form ajax handling with button instead
    $('#submitform').click(function (e) {
        ajaxOnSubmit();  // DEFINE THIS FUNC
        $.ajax({
                type: "POST",
                url: "/pred_endpoint",
                data: {message: $("#message").val(), 
                    model:  $("#model-select").val(),
                    filename: filename, 
                    group_result: $("#results-select").val()},
        beforeSend: function(){
        // Show image container
        $("#loader").show()},
        success: function (data) {
            ajaxCallback(data),
            $("#loader").hide();
            $("#message").show();
            document.getElementById('multiFiles').value=null;
            filename = ""
        },
            error: function(ajaxResponse, errorStr) {
                ajaxOnError(ajaxResponse, errorStr);  // DEFINE THIS FUNC
            },
            timeout: 90*1000
        });
        e.preventDefault();
    });
});

$(document).ready(function (e) {
    $('#upload').on('click', function () {
        var form_data = new FormData();
        var ins = document.getElementById('multiFiles').files.length;

        if(ins == 0) {
            filename = "";
            $('#msg').html('<span style="color:red">Select at least one file</span>');
            return;
        }

        if(ins > 1) {
            filename = "";
            $('#msg').html('<span style="color:red">Select only one file</span>');
            return;
        }

        for (var x = 0; x < ins; x++) {
            form_data.append("files[]", document.getElementById('multiFiles').files[x]);
        }

        $.ajax({
            url: 'python-flask-files-upload', // point to server-side URL
            dataType: 'json', // what to expect back from server
            cache: false,
            contentType: false,
            processData: false,
            data: form_data,
            type: 'post',
            success: function (response) { // display success response
                filename = response.filename ;
                clearBox('msg');
                $('#msg').append(response.message + '<br/>');
                clearBox('info_box');
                $(".info_box").append(`Will evaluate: ` + filename.bold());
                document.getElementById('message').value = "";
                $("#message").hide()
                },
            error: function (response) {
                clearBox('msg');
                // $('#msg').html(response.message); // display error response
                $('#msg').append(response['responseJSON'].message);
                return;
            }
        });
    });
});


$(document).ready(function() {
    document.getElementById('model-select').addEventListener("input", function (evt) {
    srt = mdl_srt[this.value];
    elm = document.getElementById('results-select');
    document.getElementById('results-select').innerHTML = ""

    for (var i = 0; i < srt.length; i++) {
        option = document.createElement("option")    
        option.text = srt[i]
        option.value = srt[i]
        document.getElementById('results-select').add(option)
    }    
    }, true);
})
