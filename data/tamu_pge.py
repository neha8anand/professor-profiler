import requests
import json
from bs4 import BeautifulSoup

faculties_page_url = "https://engineering.tamu.edu/profile-data.json"
faculties_page = requests.get(faculties_page_url)

faculty_names_dictionary = {}

for faculty_page in faculties_page.json():
    if 'petroleum' in faculty_page['tag']:
        faculty = faculty_page['name']
        link = "https://engineering.tamu.edu" + faculty_page['link']
        phone = faculty_page['phone']
        office = faculty_page['office']
        email = faculty_page['email']
        title = faculty_page['titles'][0]

        faculty_names_dictionary[faculty] = {'title' : title,
                                            'link' : link,
                                             'phone' : phone,
                                            'office': office,
                                            'email': email}

pge_dictionary = {}
pge_dictionary['university_name'] = 'Texas A&M University--College Station'
pge_dictionary['faculty_names'] = faculty_names_dictionary

with open('tamu_pge.json', 'w') as tamu_pge_faculty_names_file:
    json.dump(pge_dictionary, tamu_pge_faculty_names_file)
