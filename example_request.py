import requests

urlapi='http://localhost:3130/api'
urlping='http://localhost:3130/ping'

# file_path = 'path/to/your/data.csv'
file_path = "upload/app-test-data.csv"
setup_path = "setup.json"

try:
    ping = requests.get(urlping).json()['Status']
    print(ping)
except:
    print(f"Inference servers is down, please try again in a bit")

try:
    if setup_path and file_path:
        files = {'eval_file': open(file_path,'r',encoding="utf-8"),'setup': open(setup_path,'r',encoding="utf-8")}
        print(f"Starting inference request")
        r = requests.post(urlapi, files=files)
    elif setup_path and not file_path:
        setup = {'setup': open(setup_path,'r',encoding="utf-8")}
        print(f"Starting inference request")
        r = requests.post(urlapi, files = setup)
    print(r.json())
except FileNotFoundError:
    print(f"File could not be found, please check the specified file and setup path and filename")


