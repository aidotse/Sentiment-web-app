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
`docker build -f Dockerfile -t project:myapp .`
This command needs to be run from the Bert-app directory 

When you have built your container you can run it with the command
`docker run -p 3130:3130 project:myapp`

When the app has launched inside the container you can access the web app in your browser of choice by writing: 
* `http://localhost:3130/`

Or through your  IPv4 Address, access it through `ipconfig` in `CMD` or the corresponding call for your os. 
* `http://IPv4 Address:3130/`

## Models:
The models used for the sentiment classification are available for download and standalone usage at:
[Fear sentiment model](https://huggingface.co/RecordedFuture/Swedish-Sentiment-Fear)    
[Violence sentiment model](https://huggingface.co/RecordedFuture/Swedish-Sentiment-Violence)
[Violence sentiment target model](https://huggingface.co/RecordedFuture/Swedish-Sentiment-Violence-Targets)
[Fear sentiment target model](https://huggingface.co/RecordedFuture/Swedish-Sentiment-Fear-Targets)
[Swedish NER model](https://huggingface.co/RecordedFuture/Swedish-NER)
