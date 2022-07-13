import json
import requests

header = {'Content-Type': 'application/json', \
                  'Accept': 'application/json'}


with open('test_json', 'r') as test_json_txt:
    test_json = test_json_txt.read()

resp = requests.post("http://localhost:5000/", \
                    json = test_json,\
                    headers= header)

print(resp)
print(resp.json())
