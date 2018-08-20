import requests
from bs4 import BeautifulSoup
import json

# school ids for lookup on ratemyprofessors.com
with open('sids.json', 'r') as sids:
    sids = json.load(sids)

# school and faculty names for lookup on ratemyprofessors.com
with open('ut_pge_profs.json', 'r') as f:
    ut_pge_profs = json.load(f)

with open('stanford_pge_profs.json', 'r') as f:
    stanford_pge_profs = json.load(f)

with open('tamu_pge_profs.json', 'r') as f:
    tamu_pge_profs = json.load(f)

# university rankings and other info pulled from usnews website
with open('university_rankings.json', 'r') as f:
    university_rankings = json.load(f)

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36'}

def rating_and_tags(sid, school_name, faculty_name):
    '''Gives rating and tags for a professor pulled from ratemyprofessors.com
       given sid (school's id as per ratemyprofessors.com), school_name and faculty_name
    '''

     # Getting rating url by searching school and faculty name
    url = f"http://www.ratemyprofessors.com/search.jsp?queryoption=HEADER&queryBy=teacherName&schoolName={school_name}&schoolID={sid}&query={faculty_name}"
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    try:
        rating_url = "http://www.ratemyprofessors.com" + soup.select("li.listing.PROFESSOR")[0].find("a")["href"]

        # Getting rating and profile tags for the rating url
        response2 = requests.get(rating_url, headers=headers)
        soup2 = BeautifulSoup(response2.content, 'html.parser')
        rating = float(soup2.select("div.grade")[0].get_text())
        tags = [tag_desc.get_text() for tag_desc in soup2.select("span.tag-box-choosetags")]

    except:
        rating, tags = 'N/A', 'N/A'

    return {'rating': rating, 'tags': tags}

# Creating complete university database
university = university_rankings.copy()
schools_for_lookup = ['University of Texas at Austin', 'Stanford University', 'Texas A&M University at College Station']
sids_for_lookup = ['1255', '953', '1003']

for sid, school in  zip(sids_for_lookup, schools_for_lookup):

    if school == 'University of Texas at Austin':
        university['petroleum-engineering']['University of Texas--Austin (Cockrell)'] ['faculty_names'] = {}
        for faculty in ut_pge_profs['faculty_names']:
            university['petroleum-engineering']['University of Texas--Austin (Cockrell)'] ['faculty_names'][faculty] = rating_and_tags(sid, school, faculty)

    if school == 'Stanford University':
        university['petroleum-engineering']['Stanford University'] ['faculty_names'] = {}
        for faculty in stanford_pge_profs['faculty_names']:
            university['petroleum-engineering']['Stanford University'] ['faculty_names'][faculty] = rating_and_tags(sid, school, faculty)

    if school == 'Texas A&M University at College Station':
        university['petroleum-engineering']['Texas A&M University--College Station'] ['faculty_names'] = {}
        for faculty in tamu_pge_profs['faculty_names']:
            university['petroleum-engineering']['Texas A&M University--College Station'] ['faculty_names'][faculty] = rating_and_tags(sid, school, faculty)

# Writing results to a json file
with open('database.json', 'w') as f:
    json.dump(university, f)
