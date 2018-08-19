import requests
from bs4 import BeautifulSoup
import json

# school ids for lookup on ratemyprofessors.com
with open('sids.json', 'r') as sids:
    sids = json.load(sids)
