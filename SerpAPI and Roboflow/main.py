import urllib.request
from serpapi import GoogleSearch
import time
#import cv2
'''
In this example we iterate over only one search query,
doing pagination until results are present, 
only allowing images with SafeSearch active, 
collecting as many results that can be found,
and extracting original size image + saving locally
'''

# import roboflow
from roboflow import Roboflow 

from pathlib import Path


def serpapi_get_google_images():
  image_results = [
  ]  # Stores locations of every image, so we can go back and download them

  search_phrases = ["airport baggage xrays"]

  for query in search_phrases:  # array with 1 item, could be expanded to query multiple searches
    # search query parameters
    params = {
      "engine":
      "google",  # search engine. Google, Bing, Yahoo, Naver, Baidu...
      "q": query,  # search query
      "tbm": "isch",  # image results
      "safe": "active",  # safe search
      "num": "100",  # number of images per page
      "ijn": 0,  # page number: 0 -> first page, 1 -> second...
      "api_key":
      "API_KEY_GOES_HERE",  # https://serpapi.com/manage-api-key
      # other query parameters: hl (lang), gl (country), etc
    }

    try:
      print("Searching...")
      assert (len(params["api_key"]) == 64
              ), "Something likely went wrong with the API key"
      search = GoogleSearch(params)  # where data extraction happens
      print("Searching completed!")
    except:
      print("Something likely went wrong with the API key")

    images_is_present = True  # boolean confirming google still has results on the page
    image_counter = 0  # counter to keep track of number of images, and stop when necessary

    while images_is_present:
      results = search.get_dict()  # JSON -> Python dictionary

      # checks for "Google hasn't returned any results for this query." and haven't exceeded arbitrary number of images
      if "error" not in results:
        # iterates through the results returned from SerpAPI, and stores every unique images location in our results list
        for image in results["images_results"]:
          if image["original"] not in image_results:
            image_results.append(image["original"])
        # update to the next page
        params["ijn"] += 1
      else:
        images_is_present = False

  # -----------------------
  # Downloading images
  for index, image in enumerate(image_results,
                                start=1):  # since results array starts at 1
    if image_counter > 20:
      break

    start = time.time()
    print(f"Downloading {index} image...")

    # telling the website what "device" is viewing it
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]

    urllib.request.install_opener(opener)

    try:  # tests if image can be scraped
      urllib.request.urlretrieve(image,
                                 f"dataset/original_size_img_{index}.jpg")
      image_counter += 1
      end = time.time()
      assert end - start < 30
      #print(end - start)
    except:  # prints error if website prevents the download or some other error occurs
      print(f"Error downloading image {index}...")

  #-------------------------
  # [OPTIONAL] Uploading to Roboflow
  # after importing the roboflow Python Package
  # directly upload the images to Roboflow's website
  #   creating the Roboflow object
  #   obtaining your API key: https://docs.roboflow.com/rest-api#obtaining-your-api-key
"""
    rf = Roboflow(api_key="13235235252335332235523235")

    # using the workspace method on the Roboflow object
    workspace = rf.workspace()

    # identifying the project for upload
    project = workspace.project("https://universe.roboflow.com/chinmay-ranganath-ohlji/outsiden-t")

    ## if you want to attempt reuploading image on failure
    for index in len(image_results):
        image = Path(f"dataset/original_size_img_{index}.jpg")
        if image.exists():
            project.upload(image, num_retry_uploads=3)
        else:
            break



if __name__ == "__main__":
  try:
    serpapi_get_google_images()
  except:
    print(
      "Things didn't get set up properly, make sure you've double checked all the place-holder fields."
    )
"""
