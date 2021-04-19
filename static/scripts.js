var result_data
filename = ""

var classification_breakpoint = 0


function clearBox(elementID){
                            document.getElementById(elementID).innerHTML = "";
                            };

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

function ajaxOnSubmit(){clearBox('results');
                        //$(".results").append( "<p class="alert alert-success"><strong>Success!</strong> This alert box could indicate a successful or positive action.")</p>}
                        $(".results").append(`Processing the data`);}
function ajaxOnError(){}
function ajaxCallback(data){clearBox('results');
                            console.log({data:data}) ;
                            console.log($("input[name='model_button']:checked").val());
                            if ($("input[name='model_button']:checked").val() == 'violence'){
                                classification_breakpoint = 0.575}
                            else if ($("input[name='model_button']:checked").val() == 'fear'){
                                classification_breakpoint = 0.5};
                            console.log(classification_breakpoint);
                            if (data.message == "no_data_uploaded_or_in_text_area_"){
                                $(".results").append(`No data available`)}
                            else{

                                if (data.pred.length > 1){
                                    //$(".results").append("<p>To many texts to show individual predictions </p>");
                                    //$(".results").append("<p>Prediction complete, results ready for download </p>")
                                    var table_data = [];
                                    for (i = 0; i < data.message.length; i++){
                                        new_line = {'message':data.message[i], 'pred':(data.pred[i]).toFixed(2)};
                                        table_data[i]=new_line
                                        };
                                    console.log({table_data:table_data});
                                    if ($('#res_table').bootstrapTable('getData').length==1) {
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
                                                }],
                                                
                                                data: table_data
                                            })
                                                }
                                        else{
                                            $('#res_table').bootstrapTable('load', table_data);
                                            }
                                        }
                                else{
                                    $(".results").html(`Sentiment prediction `);
                                    $(".results").append(data.pred);
                                    };
                                    $(".results").append(`<br/>`);

                                if(data.pred.length == 1){
                                    if (data.pred > classification_breakpoint){
                                        $(".results").append(`Text classified as `)}
                                    else {
                                        $(".results").append(`Text classified as NON `)};
                                    $(".results").append($("input[name='model_button']:checked").val())};
                                result_data = data;
                                }
                            }

function exportJson(el) {
                            var data = "text/json;charset=utf-8," + encodeURIComponent(JSON.stringify({"message":result_data.message,"prediction":result_data.pred}));
                            // what to return in order to show download window?

                            el.setAttribute("href", "data:"+data);
                            el.setAttribute("download", "results.json");}

function exportCSV(el) {    console.log(result_data);
                            var csv_string = "";
                            for (i = 0; i < result_data.message.length; i++){
                                tt = result_data.message[i].replace(/,/g,';') // replace commas with semi-commas (, -> ;) to avoid erros in the csv
                                csv_string += `${tt},${result_data.pred[i]}\r\n`}
                            console.log(csv_string)

                            var csv = csv_string
                            console.log(csv)

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



// {#  Modified version of ajax_form_submit for two buttons  #}
$(document).ready(function() {
    // Hide user select form when job is changed
    let selectForm = $('#user-form');
    // User Submit form ajax handling with button instead
    $('#submitform').click(function (e) {
        // let url = "{{ url_for('.predict') }}";
        ajaxOnSubmit();  // DEFINE THIS FUNC
        $.ajax({
            type: "POST",
            url: "/predict",
            data: {message: $("#message").val(), 
            model:  $("#model-select").val(),
            filename: filename, 
            group_result: $("#results-select").val()},
            beforeSend: function(){
            // Show image container
            $("#loader").show()},
            success: function (data) {
                ajaxCallback(data),
                $("#loader").hide();   // DEFINE THIS FUNC
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
                $('#msg').append(response.message + '<br/>')
                },
            error: function (response) {
                clearBox('msg');
                // $('#msg').html(response.message); // display error response
                $('#msg').append('<span style="color:red">File needs to be in CSV format</span>');
                return;
            }
        });
    });
});