import os
import socket
import json
import numpy as np
import pandas as pd
import torch
from flask import Flask, render_template, request, jsonify
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizerFast, BertForTokenClassification
from werkzeug.utils import secure_filename


def next_free_port(port=3130, max_port=65535):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while port <= max_port:
        try:
            sock.bind(('', port))
            sock.close()
            return port
        except OSError:
            port += 1
    raise IOError('no free ports')


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])  #
        return item

    def __len__(self):
        return len(self.labels)


def load_sentiment_classifier(model):
    classifier = BertForSequenceClassification.from_pretrained(
        # "KB/bert-base-swedish-cased", # Use the 12-layer BERT model, with a cased vocab.
        model,
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    return classifier
def load_token_classifier(model):
    classifier = BertForTokenClassification.from_pretrained(
        # "KB/bert-base-swedish-cased", # Use the 12-layer BERT model, with a cased vocab.
        model,
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    return classifier


def tokenize_data(data, tokenizer):
    label = np.zeros(len(data))  # dummy labels are used so that the entire Data loader class needs to be rewritten, does not affect classification
    label = torch.tensor(label, dtype=int)
    encodings = tokenizer(list(data), truncation=True, padding=True, max_length= 512)
    transformed_data = Dataset(encodings, label)
    return transformed_data


def pred_frag(tokenized_data, classifier):
    pred = np.array([])
    ver_data_loader = DataLoader(tokenized_data, batch_size=2, shuffle=False)
    for batch in tqdm(ver_data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        # _, logits = classifier(**batch)  # run local
        output = classifier(**batch)  # run MLab
        logits = output.logits  # run MLab
        p_soft_max = torch.softmax(logits, dim=1)[:, 1:].tolist()
        # tmp_pred = p_soft_max[0] # non scaled version
        tmp_pred = [p_soft_max[0][0] * 0.75, p_soft_max[0][1]]  # scaled weak sentiment from [0 1] to [0 0.75]
        pred = np.append(pred, tmp_pred)
        return pred


def save_files(file):
    errors = {}
    success = False
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        success = True
    else:
        errors['message'] = 'File extension is not allowed'

    if success and errors:
        errors['message'] = 'File successfully uploaded'
        resp = jsonify(errors)
        resp.status_code = 206
        return resp
    if success:
        resp = jsonify({'message': 'File successfully uploaded', 'filename': filename})
        resp.status_code = 201
        return resp
    else:
        resp = jsonify(errors)
        resp.status_code = 400
        return resp

UPLOAD_FOLDER = 'upload'
ALLOWED_EXTENSIONS = {'csv'}

if not os.path.exists("upload"):
    os.makedirs("upload")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # max input file size is roughly 10mb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Loading tokenizer")
tokenizer = BertTokenizerFast.from_pretrained("RecordedFuture/Swedish-Sentiment-Fear")
print(f"Loading Violence Sentiment model")
classifier_violence = load_sentiment_classifier("RecordedFuture/Swedish-Sentiment-Violence").to(device)
print(f"Loading Fear Sentiment model")
classifier_fear = load_sentiment_classifier("RecordedFuture/Swedish-Sentiment-Fear").to(device)
print(f"Loading Fear sentiment target model")
classifier_fear_targets = load_token_classifier("RecordedFuture/Swedish-Sentiment-Fear-Targets").to(device)
print(f"Loading Violence sentiment target model")
classifier_violence_targets = load_token_classifier("RecordedFuture/Swedish-Sentiment-Violence-Targets").to(device)

def model_selector(setup):
    if setup["model"] == "fear": # make the sentiment classifier selection
        classifier = classifier_fear
    elif setup["model"] == "violence":
        classifier = classifier_violence
    elif setup["model"] == "fear_target":
        classifier = classifier_fear_targets
    elif setup["model"] == "violence_target":
        classifier = classifier_violence_targets
    return classifier

def input_source(setup):
    if setup['message']: # the text box is used
        message = setup["message"]
    elif setup['filename'] and not setup['message']:
        message = pd.read_csv(f"../Bert-app/upload/{setup['filename']}", header=None, usecols=[0])
        # read the file name from the upload folder, use the first col as the data column
    else:
        return {"message": "No data received in payload", "pred": ""}
    return message

def prepare_data(setup, message):

    index_all = []  # var for storing the indexing
    index = 0  # start indexing at zero

    if setup['message']:  # if the text message box is used it should take priority
        data_pred = []
        s_frag = message.split(".")  # split all fragments on ".", if "." not in frag nothing happens
        for s in s_frag:  # loop through the list of split strings
            if s != "" and s != " ":  # if a string is not whitespace save it to data for eval
                data_pred.append(s)
                index_all.append(index)  # append the indexing for max sorting
        message = pd.Series(message)
    elif not setup['message'] and setup["filename"]:  # double check to make sure that the tex box is not in use and a file is uploaded
        data_pred = []
        # message = pd.read_csv(f"../Bert-app/upload/{setup['file']}", header=None, usecols=[
        #     0])  # read the file name from the upload folder, use the first col as the data column
        if len(message) > 1:  # different methods for handling if all the data is present in one csv cell or not, due to the list() method transorming each char to a seperate string in that case
            tmp_data = list(message.squeeze())  # transform DF to Series and format it as a list
        else:
            tmp_data = [message.squeeze()]
        for frag in tmp_data:  # roll through all fragments from the file
            s_frag = frag.split(".")  # split all fragments on ".", if "." not in frag nothing happens
            for s in s_frag:  # loop through the list of split strings
                if s != "" and s != " ":  # if a string is not whitespace save it to data for eval
                    data_pred.append(s)
                    index_all.append(index)  # indexing for max sorting
            index += 1  # inc index after one cell is processed.
    else:
        print("No data in text window and no file uploaded")
        return {"message": "no_data_uploaded_or_in_text_area_", "pred": 0}

    return data_pred, index_all

def predict(setup):

    message = input_source(setup)

    classifier = model_selector(setup)

    data_pred , index_all = prepare_data(setup, message)

    pred = [] # var for storing the predctions
    label = [] # var for storing the labels of the
    batches = chunks(data_pred, 1)  # can probably batch it in larger than 1
    for batch in batches:
        tokenized_data = tokenize_data(batch, tokenizer)
        tmp_pred = pred_frag(tokenized_data, classifier=classifier)
        pred.append(round(np.max(tmp_pred), 2))
        label.append(np.argmax(tmp_pred))

    ### all post processesing of the results should be done in the "front end".
    ### the back end should only do the prediction and always return the results in the same format
    data_disp = []
    pred_disp = []
    if setup['group_result'] == 'unsorted':  # if data aggregation button selection is seperate just continue, all if formatted correctly already
        data_disp = data_pred
        pred_disp = pred

    elif setup['group_result'] == 'sorted':  # if the button is set to max
        if len(pred) < 2:  # if len of pred is 1 then just continue, you cant sort a single float
            data_disp = data_pred
            pred_disp = pred
        else:  # if the number of predictions made is higher than 5, get the indexes of the top 5 predictions and get the pred values and fragments
            sorted_based_max_pred = np.array(pred).argsort()[:][::-1]  # sorted
            pred_disp = [pred[i] for i in sorted_based_max_pred] # format the predictions to make the output consistent
            data_disp = [data_pred[i] for i in sorted_based_max_pred] # format the data to make the output consistent
    elif setup['group_result'] == 'max':  # if the button is set to max
        if len(pred) < 2: # if len of pred = 1 then do nothing since the max of a float is itself
            data_disp = data_pred
            pred_disp = pred
        else:
            pred_max = []
            for uniq in np.unique(index_all):  # check all unique indexes
                tmp_pred = []
                for i, val in enumerate(index_all):  # check all available indexes from splitting
                    if val == uniq:  # get attribute prediction with the same index
                        tmp_pred.append(pred[i])
                pred_max.append(np.max(tmp_pred))
            pred_disp = pred_max

            if len(message) > 1 and type(message).__name__ == 'DataFrame':  # different methods for handling if all the data is present in one csv cell or not, due to the list() method transorming each char to a seperate string in that case
                data_disp = list(message.squeeze())  # transform DF to Series and format it as a list
            else:
                data_disp = [message.squeeze()]


    ret = {"message": data_disp,
           "pred": pred_disp,
           "message_raw": data_pred,
           "pred_raw": pred,
           "index": index_all}

    return ret

@app.route('/ping', methods = ["GET"])
def ping():
    return jsonify({"Status":"Server is live"})

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/guide')
def guide():
    return render_template('guide.html')

@app.route('/echo/<message>',methods = ["GET"])
def echo(message):
    print(f"{message}")
    return {"message":message}

@app.route('/api', methods=["POST"])
def api():

    setup = request.files['setup'].read().decode("utf-8")
    setup = json.loads(setup)

    if 'eval_file' in request.files: # check if "eval_file" is in request.files, only occures when the "eval_file" input is used in a curl request
        file = request.files['eval_file'] # take the uploaded file
        resp = save_files(file) # run it through the save_file function to save it in the upload folder
        resp = resp.json # read the response as JSON
        setup['filename'] = resp['filename']
    else:
        setup['filename']= ""

    resp = predict(setup)
    return resp

@app.route('/api/input',methods = ["GET"])
def api_input():
    return jsonify({"group_result":["unsorted","sorted","max"],
                    "model":["fear","violence"],
                    "message": "any string",
                    "eval_file": "@path/to/file.csv"
                    })

@app.route('/pred_endpoint', methods=["POST"])
def pred_endpoint():
    setup = {
        "group_result": request.form['group_result'],
        "model": request.form['model'],
        "filename": request.form['filename'],
        "message": request.form['message']
    }

    response = predict(setup)

    return response

@app.route('/python-flask-files-upload', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({'message': 'No file part in the request'})
        resp.status_code = 400
        return resp

    files = request.files.getlist('files[]')
    for file in files:
        resp = save_files(file)

    return resp


if __name__ == "__main__":
    # port = next_free_port()
    port = 3130
    l_host = f"http://localhost:{port}/"
    ipv4 = f"http://{socket.gethostbyname(socket.gethostname())}:{port}/"
    print(f"\n\n"
          f"# Web app is hosted on port: {port}\n"
          f"# To access the app go to\n"
          f"# {l_host}\n"
          f"# Or\n"
          f"# {ipv4}\n \n")
    print(f"The link below is broken, follow the above steps to access the web app")
    app.run(debug=False, host='0.0.0.0', port=port)

