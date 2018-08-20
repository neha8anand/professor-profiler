import requests
import json
from bs4 import BeautifulSoup

faculties_page_url = 'https://engineering.tamu.edu/profile-data.json'
faculties_page = requests.get(faculties_page_url)

faculty_names_dictionary = {}

for faculty_entry in faculties_page.json():
    if 'petroleum' in faculty_entry['tag']:
        faculty = faculty_entry['name']
        title = faculty_entry['titles'][0]
        page = 'https://engineering.tamu.edu' + faculty_entry['link']
        phone = faculty_entry['phone']
        office = faculty_entry['office']
        email = faculty_entry['email']
        research_areas = '' # for faculties whose research_areas are not defined
        google_scholar_link = ''

        # google-scholar profile link
        faculty_page = requests.get(page)
        soup = BeautifulSoup(faculty_page.content, 'html.parser')
        for a_tag in soup.select('a.button '):
            if 'scholar.google.com' in a_tag['href']:
                google_scholar_link = a_tag['href']
                break

        # research areas
        for div_tag in soup.select('div.simple-list-collection__list'):
            if 'Research Interests' in div_tag.get_text():
                research_areas = div_tag.select("ul.no-bullets")[0].get_text().strip('\n').strip('\xa0').split('\n')
                research_areas = [area for area in research_areas if area]
                break

        faculty_names_dictionary[faculty] = {'title' : title,
                                             'page' : page,
                                             'phone' : phone,
                                             'office': office,
                                             'email': email,
                                             'research_areas' : research_areas,
                                             'google_scholar_link' : google_scholar_link}

pge_dictionary = {}
pge_dictionary['university_name'] = 'Texas A&M University at College Station'
pge_dictionary['faculty_names'] = faculty_names_dictionary

with open('tamu_pge.json', 'w') as tamu_pge_faculty_names_file:
    json.dump(pge_dictionary, tamu_pge_faculty_names_file)
