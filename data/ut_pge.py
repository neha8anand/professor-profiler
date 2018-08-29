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

pge_dictionary = {}
pge_dictionary['university_name'] = "University of Texas at Austin"

faculty_names_dictionary = {}

for pge_faculty_entry in pge_faculty_entries:
        pge_faculty_info = list(pge_faculty_entry.find_all("td"))

        # getting the faculty page url
        pge_faculty_link = pge_faculty_info[0].find("a")["href"]
        pge_faculty_page_url = pge_url + pge_faculty_link
        pge_faculty_page = requests.get(pge_faculty_page_url, verify=False)
        pge_faculty_soup = BeautifulSoup(pge_faculty_page.content, "html.parser")

        # extracting name, title, phone, office, email, research_areas
        faculty = pge_faculty_soup.select("div#mainbody2")[0].find_all("h1")[0].get_text()
        faculty = faculty.replace("  ", " ") # Name in HTML contains 2 spaces instead of 1
        title = pge_faculty_soup.select_one("p.depttitle").get_text()

        facdata = pge_faculty_soup.select_one("div.facdata").select("p")
        phone = facdata[-1].get_text().split('\n')[0].split(': ')[-1]
        office = facdata[-1].get_text().split('\n')[1].split(': ')[1]
        email = facdata[-1].get_text().split('\n')[0].split(': ')[1].split('Phone')[0]

        for p_tag in pge_faculty_soup.select("p"):
            if 'Research Areas' in p_tag.get_text():
                research_areas = p_tag.get_text().split(': ')[-1].split(', ')

        # google-scholar link
        google_scholar_link = ''
        for a_tag in pge_faculty_soup.select("a"):
            try:
                if 'scholar.google.com' in a_tag['href']:
                    google_scholar_link = a_tag['href']
            except:
                 google_scholar_link = ''

        faculty_names_dictionary[faculty] = {'title' : title,
                                             'page' : pge_faculty_page_url,
                                             'phone' : phone,
                                             'office': office,
                                             'email': email,
                                             'research_areas' : research_areas,
                                             'google_scholar_link' : google_scholar_link}

pge_dictionary['faculty_names'] = faculty_names_dictionary

with open('json/ut_pge.json', 'w') as ut_pge_faculty_names_file:
    json.dump(pge_dictionary, ut_pge_faculty_names_file)
