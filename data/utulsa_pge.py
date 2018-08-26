import requests
import bs4
import json
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings("ignore") # utulsa's certificate is not trusted

pge_url = "https://faculty.utulsa.edu"
pge_faculties_page_url = pge_url + "/department/mcdougall-school-of-petroleum-engineering"
pge_faculties_page = requests.get(pge_faculties_page_url, verify=False)
pge_faculties_soup = BeautifulSoup(pge_faculties_page.content, 'html.parser')
pge_faculty_entries = pge_faculties_soup.select("tr.rowHyperlink")

pge_dictionary = {}
pge_dictionary['university_name'] = "The University of Tulsa"

faculty_names_dictionary = {}

for pge_faculty_entry in pge_faculty_entries:
        # getting the faculty page url
        pge_faculty_link = pge_faculty_entry.find("a")["href"]
        pge_faculty_page_url = pge_url + pge_faculty_link
        pge_faculty_page = requests.get(pge_faculty_page_url, verify=False)
        pge_faculty_soup = BeautifulSoup(pge_faculty_page.content, "html.parser")

        # extracting name, title, phone, office, email, research_areas
        faculty = pge_faculty_soup.select_one("h1.entry-title").get_text().split('\n', 1)[0]
        title = pge_faculty_soup.select_one("span.title-item").get_text().strip()

        phone = pge_faculty_soup.select_one("a.tel").get_text()
        email = pge_faculty_soup.select_one("div.col-sm-6").find_all("a")[1].get_text()

        research_areas_div = pge_faculty_soup.select_one("div#researchInterests")
        research_areas = []
        if research_areas_div:
            research_areas = [research_area for research_area in research_areas_div.select_one("p").get_text().split('\n') if research_area != ""]

        faculty_names_dictionary[faculty] = {'title' : title,
                                             'page' : pge_faculty_page_url,
                                             'phone' : phone,
                                             'office': '',
                                             'email': email,
                                             'research_areas' : research_areas,
                                             'google_scholar_link' : ''}

pge_dictionary['faculty_names'] = faculty_names_dictionary

with open('utulsa_pge.json', 'w') as ut_pge_faculty_names_file:
    json.dump(pge_dictionary, ut_pge_faculty_names_file)
