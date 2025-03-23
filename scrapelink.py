import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


def scrapeLink(url):
    # Send a GET request to the website
    response = requests.get(url)

    # If the request is successful (status code 200)
    if response.status_code == 200:
        # Parse the content of the webpage
        soup = BeautifulSoup(response.content, "html.parser")

        # print(soup.get_text())

        # # Example: Extract all the links (anchor tags)
        # links = soup.find_all("a")

        # # Print all the links
        # for link in links:
        #     print(link.get("href"))
        return soup.get_text()
    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
        return " "


def scrapeLinks(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        links = [urljoin(url, a.get("href")) for a in soup.find_all("a", href=True)]
        return links
    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
        return []
