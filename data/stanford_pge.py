import requests
import bs4
import json
from bs4 import BeautifulSoup

def extract_from_contact(contact):
    '''Extracts office, phone and email from contact'''

    info = []

    for elem in contact:
        if type(elem) != bs4.element.Tag:
            info.append(elem.strip())

    office, phone, email = '', '', ''

    for elem in info:
        if '@' in  elem:
            email = elem
        elif '(' in elem:
            phone = elem
        elif elem:
            office = elem

    return phone, office, email

pge_faculties_url = "https://pangea.stanford.edu/ere/people/all?tmp_associate_type=faculty&field_ses_phd_student_value=All&name="
pge_faculties = requests.get(pge_faculties_url)
pge_faculties_soup = BeautifulSoup(pge_faculties.content, 'html.parser')
pge_faculty_names = [name.get_text().strip('\n').strip(' ') for name in pge_faculties_soup.find_all(class_="views-field views-field-name ses-people-column-name")]
pge_faculty_titles = [title.get_text().strip('\n').strip(' ') for title in pge_faculties_soup.select("td.views-field.views-field-field-ses-title-override")]
pge_faculties_pages = [entry.select_one('a')['href'] for entry in pge_faculties_soup.find_all(class_="views-field views-field-name ses-people-column-name")]
pge_faculties_pages_url = ['https://pangea.stanford.edu' + entry for entry in pge_faculties_pages]
pge_faculty_research_areas = [area.get_text().strip('\n').strip(' ') for area in pge_faculties_soup.select("td.views-field.views-field-field-ses-research-override")]

# cleaning up the contact info and dividing it in office, phone and email categories
pge_faculty_contact = [info.children for info in pge_faculties_soup.select("td.views-field.views-field-field-office-location.ses-people-column-contact")]
pge_faculty_extracted_from_contact = [extract_from_contact(contact) for contact in pge_faculty_contact]

faculty_names_dictionary = {}

for i, faculty in enumerate(pge_faculty_names):
    faculty_names_dictionary[faculty] = {'title' : pge_faculty_titles[i],
                                         'page' : pge_faculties_pages_url[i],
                                         'phone' : pge_faculty_extracted_from_contact[i][0],
                                         'office': pge_faculty_extracted_from_contact[i][1],
                                         'email': pge_faculty_extracted_from_contact[i][2],
                                         'research_areas': pge_faculty_research_areas[i]
                                         }

pge_dictionary = {}
pge_dictionary['university_name'] = 'Stanford University'
pge_dictionary['faculty_names'] = faculty_names_dictionary

with open('stanford_pge.json', 'w') as st_pge_faculty_names_file:
    json.dump(pge_dictionary, st_pge_faculty_names_file)
