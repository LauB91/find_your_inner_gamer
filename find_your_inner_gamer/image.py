import requests
from bs4 import BeautifulSoup

def get_img(url):
    response = requests.get(url).text
    soup = BeautifulSoup(response, "html.parser")
    try:
        return soup.find('img', class_='game_header_image_full').attrs['src']
    except AttributeError:
        return 'no image'
