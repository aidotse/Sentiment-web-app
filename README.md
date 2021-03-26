## User guide to the app
A user guide to the web app is available in the app. It is essentially the guide.txt with a bit of formatting. 

## Run 
The web app can be run by pulling the the repo. 

Start with installing the Python dependencies:
`pip install -r requirements.txt`

When the dependencies are installed launch the app from the directory with the command:
`python app.py`

The app will launch and start downloading the Bert models from huggingface, this step will take a while but should only need to be done once. 

The app will be available at: 
* `http://localhost:3130/`
Or through your  IPv4 Address in your browser of choice, access IPv4 Address by passing `ipconfig` in `CMD` or the corresponding call for your os. 
* `http://IPv4 Address:3130/`
  
## Docker: 
#### Build
The web app can be packaged into a docker container using the included Docker file using 
`docker build --tag name:tag` 

### Pull from hub
This application is available from docker hub the latest tag is the version currently available on Github. 

Pull the repo using the command 
* `docker pull rffmoller/swedish-bert-web-app`
No tag needs to be included, it reverts to the latest.

to run the container use the following call: 
* `docker run -p 3130:3130 rffmoller/swedish-bert-web-app`

The `-p 3130:3130` input will map the web app which always will be hosted on port 3130 in the docker container to your local port 3130. (local port: docker port, you can use what you want for the local port)

When the app has launched inside the container you can access the web app in your browser of choice by writing: 
* `http://localhost:3130/`

Or through your  IPv4 Address, access it through `ipconfig` in `CMD` or the corresponding call for your os. 
* `http://IPv4 Address:3130/`

## Models:
The models used for the sentiment classification are available for download and standalone usage at:
[Fear model](https://huggingface.co/RecordedFuture/Swedish-Sentiment-Fear)    
[Violence model](https://huggingface.co/RecordedFuture/Swedish-Sentiment-Violence)

Before running. Make sure you have a valid Bert model directory inside of the "Bert-app" dir. 
Change the name in the "load_classifier" function in app.py 

Except for this the app should be able to run on any system
