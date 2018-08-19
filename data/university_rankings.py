import requests
from bs4 import BeautifulSoup
import json

def create_ranks(scores):
    '''
    Creates ranks corresponding to the scores, allowing for ties between universities.
    '''
    ranks = list(range(1, len(scores) + 1))
    # Allowing for ties
    for i, score in enumerate(scores[1:], start=1):
        if score == scores[i - 1]:
            ranks[i] = ranks[i-1]
    return ranks

majors = ['aerospace', 'biological-agricultural', 'biomedical','chemical-engineering',
          'civil-engineering', 'computer-engineering','electrical-engineering',
          'environmental-engineering', 'industrial-engineering','material-engineering',
          'mechanical-engineering', 'nuclear-engineering', 'petroleum-engineering']

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36'}

results = {}

for major in majors:
    results[major] = {}

    url = "https://www.usnews.com/best-graduate-schools/search?program=top-engineering-schools&name=&specialty=" + major
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Names of universities
    universities = []
    for a_tag in soup.select("a.Anchor-s8bzdzo-0.fwxkXI"):
        universities.append(a_tag.get_text())

    # locations
    locations = []
    for p_tag in soup.select("p.Paragraph-fqygwe-0.cPvbgl"):
        locations.append(p_tag.get_text())

    # scores
    scores = []
    for p_tag in soup.select("p.fqygwe-0-Paragraph-hHEPzZ.kkoztb"):
        if p_tag.get_text() != 'N/A':
            scores.append(float(p_tag.get_text()))

    # ranks (recreated from scores)
    ranks = create_ranks(scores)

    # Dropping universities with empty scores
    universities = universities[:len(scores)]
    locations = locations[:len(scores)]

    # Storing results
    for university, location, score, rank in zip(universities, locations, scores, ranks):
        results[major][university] = {'location': location, 'score' : score, 'rank': rank}

# Writing results to a json file
with open('university_rankings.json', 'w') as university_rankings:
    json.dump(results, university_rankings)
