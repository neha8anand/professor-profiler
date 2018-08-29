"""
Module containing webscraping code for ratemyprofessors.com. When run as a module, this will scrape 
ratings/reviews for professors from ratemyprofessors.com and append the results to the university
rankings.json file and write the results to the disk as majors_database.json.
"""

import requests
from bs4 import BeautifulSoup
import json

# school ids for lookup on ratemyprofessors.com
with open('json/sids.json', 'r', encoding='utf-8') as sids:
    sids = json.load(sids)

# school and faculty names for lookup on ratemyprofessors.com
with open('json/ut_database.json', 'r', encoding='utf-8') as f:
    ut_pge_profs = json.load(f)

with open('json/stanford_database.json', 'r', encoding='utf-8') as f:
    stanford_pge_profs = json.load(f)

with open('json/tamu_database.json', 'r', encoding='utf-8') as f:
    tamu_pge_profs = json.load(f)

with open('json/utulsa_database.json', 'r', encoding='utf-8') as f:
    utulsa_pge_profs = json.load(f)

# majors rankings and other info pulled from usnews website
with open('json/university_rankings.json', 'r', encoding='utf-8') as f:
    university_rankings = json.load(f)

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36'}

def rating_and_tags(sid, school_name, faculty_name):
    '''Returns rating and tags for a professor pulled from ratemyprofessors.com
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
        rating, tags = '', []

    return {'rating': rating, 'tags': tags}

# Creating complete majors database
majors = university_rankings.copy()
schools_for_lookup = ['University of Texas at Austin', 'Stanford University', 'Texas A&M University at College Station', 'University of Tulsa']
sids_for_lookup = ['1255', '953', '1003', '3963']

for sid, school in  zip(sids_for_lookup, schools_for_lookup):

    if school == 'University of Texas at Austin':
        majors['petroleum-engineering']['University of Texas--Austin (Cockrell)'] ['faculty_names'] = {}
        for faculty in ut_pge_profs['faculty_names']:
            majors['petroleum-engineering']['University of Texas--Austin (Cockrell)'] ['faculty_names'][faculty] = ut_pge_profs['faculty_names'][faculty]
            majors['petroleum-engineering']['University of Texas--Austin (Cockrell)'] ['faculty_names'][faculty].update(rating_and_tags(sid, school, faculty))
    
    if school == 'Stanford University':
        majors['petroleum-engineering']['Stanford University'] ['faculty_names'] = {}
        for faculty in stanford_pge_profs['faculty_names']:
            majors['petroleum-engineering']['Stanford University'] ['faculty_names'][faculty] = stanford_pge_profs['faculty_names'][faculty]
            majors['petroleum-engineering']['Stanford University'] ['faculty_names'][faculty].update(rating_and_tags(sid, school, faculty))
            

    if school == 'Texas A&M University at College Station':
        majors['petroleum-engineering']['Texas A&M University--College Station'] ['faculty_names'] = {}
        for faculty in tamu_pge_profs['faculty_names']:
            majors['petroleum-engineering']['Texas A&M University--College Station'] ['faculty_names'][faculty] = tamu_pge_profs['faculty_names'][faculty]
            majors['petroleum-engineering']['Texas A&M University--College Station'] ['faculty_names'][faculty].update(rating_and_tags(sid, school, faculty))

    if school == 'University of Tulsa':
        majors['petroleum-engineering']['University of Tulsa'] ['faculty_names'] = {}
        for faculty in utulsa_pge_profs['faculty_names']:
            majors['petroleum-engineering']['University of Tulsa'] ['faculty_names'][faculty] = utulsa_pge_profs['faculty_names'][faculty]
            majors['petroleum-engineering']['University of Tulsa'] ['faculty_names'][faculty].update(rating_and_tags(sid, school, faculty))

# Writing results to a json file
with open('json/majors_database.json', 'w', encoding='utf-8') as f:
    json.dump(majors, f)