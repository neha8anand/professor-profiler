import requests
import bs4
import json
from bs4 import BeautifulSoup


pge_faculties_page_url = "https://pangea.stanford.edu/ere/people/all?tmp_associate_type=faculty&field_ses_phd_student_value=All&name="
pge_faculties_page = requests.get(pge_faculties_page_url)
pge_faculties_soup = BeautifulSoup(pge_faculties_page.content, 'html.parser')
pge_faculty_names = [name.get_text().strip('\n').strip(' ') for name in pge_faculties_soup.find_all(class_="views-field views-field-name ses-people-column-name")]
pge_faculty_titles = [title.get_text().strip('\n').strip(' ') for title in pge_faculties_soup.select("td.views-field.views-field-field-ses-title-override")]
pge_faculty_research_areas = [area.get_text().strip('\n').strip(' ') for area in pge_faculties_soup.select("td.views-field.views-field-field-ses-research-override")]
pge_faculty_contact = [info.get_text().strip('\n').strip(' ') for info in pge_faculties_soup.select("td.views-field.views-field-field-office-location.ses-people-column-contact")]

faculty_names_dictionary = {}

for i, faculty in enumerate(pge_faculty_names):
    faculty_names_dictionary[faculty] = {'title' : pge_faculty_titles[i],
                                        'research_areas': pge_faculty_research_areas[i],
                                        'contact': pge_faculty_contact[i]}

pge_dictionary = {}
pge_dictionary['university_name'] = 'Stanford University'
pge_dictionary['faculty_names'] = faculty_names_dictionary

with open('stanford_pge.json', 'w') as st_pge_faculty_names_file:
    json.dump(pge_dictionary, st_pge_faculty_names_file)
