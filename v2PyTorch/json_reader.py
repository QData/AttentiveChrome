import json


with open('files.json') as f:
    data = json.load(f)


with open('model.tsv','w') as f: 
	for file in data['files']:
		if '.pt' in file['links']['self']:
			f.write(file['links']['self']+'?download=1'+'\t')
			f.write(file['checksum'].split('md5:')[1]+'\n')
