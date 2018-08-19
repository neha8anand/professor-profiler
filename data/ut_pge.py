import requests
import bs4
import json
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings("ignore") # ut's certificate is not trusted

pge_url = "https://www.pge.utexas.edu"
pge_faculties_page_url = pge_url + "/facultystaff/profiles"
pge_faculties_page = requests.get(pge_faculties_page_url, verify=False)
pge_faculties_soup = BeautifulSoup(pge_faculties_page.content, 'html.parser')

pge_faculty_entries = pge_faculties_soup.find_all(class_="facentry")

faculty_names_dictionary = {}
faculty_names_dictionary['university_name'] = "University of Texas at Austin"
faculty_names_dictionary['faculty_names'] = []

for pge_faculty_entry in pge_faculty_entries:
        pge_faculty_info = list(pge_faculty_entry.find_all("td"))
        # getting the faculty page url
        pge_faculty_link = pge_faculty_info[0].find("a")['href']

        pge_faculty_page_url = pge_url + pge_faculty_link
        pge_faculty_page = requests.get(pge_faculty_page_url, verify=False)

        pge_faculty_soup = BeautifulSoup(pge_faculty_page.content, 'html.parser')
        pge_faculty_name = pge_faculty_soup.select("div#mainbody2")[0].find_all("h1")[0].get_text()

        faculty_name = pge_faculty_name.replace("  ", " ") # Name in HTML contains 2 spaces instead of 1
        faculty_names_dictionary['faculty_names'].append(faculty_name)

with open('ut_pge.json', 'w') as ut_pge_faculty_names_file:
    json.dump(faculty_names_dictionary, ut_pge_faculty_names_file)
