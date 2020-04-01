import csv
import requests
import shutil
import time
from pathlib import Path
from os import path
from PIL import Image
import pandas as pd
from tempfile import NamedTemporaryFile

img_dict = {}

with open('data_clean.csv', mode='r', encoding="latin-1") as csv_file, open('data_clean_full.csv', mode='w+', encoding="latin-1") as csv_wfile:
	csv_reader = csv.DictReader(csv_file)
	csv_writer = csv.DictWriter(csv_wfile, fieldnames=["AUTHOR", "BORN-DIED", "TITLE", "DATE", "TECHNIQUE", "LOCATION", "URL", "FORM", "TYPE", "SCHOOL", "TIMELINE", "RAW_URL", "SPLIT_PATH", "PATH", "ARTIST_RAW", "INITIAL"])
	csv_writer.writeheader()
	line_count = 0
	for row in csv_reader:
		line_count += 1
		# if (line_count > 75):
		# 	break

		#URL String Creator
		url_string = row["URL"]
		#cut_loc = url_string.find("html")
		cut_loc = 19
		#print(url_string)
		#dl_url_start = "https://www.wga.hu/detail/"
		url_prehtml = url_string[:cut_loc]
		url_posthtml = url_string[cut_loc + len("html"):]

		url_prehtml = url_prehtml + "detail"
		
		url_posthtml = url_posthtml[:-4]
		url_posthtml = url_posthtml + "jpg"

		img_tag = url_posthtml[3:]
		tag_loc = img_tag.find("/")
		while (tag_loc != -1):
			img_tag = img_tag[tag_loc + 1:]
			tag_loc = img_tag.find("/")

		#print("Image Tag: " + img_tag)

		dl_url = url_prehtml + url_posthtml
		#print(dl_url)

		#File Path Generator
		url_post = url_posthtml[1:]
		url_1_slash = url_post.find("/")
		url_1 = url_post[url_1_slash:]
		url_2_slash = url_1[1:].find("/") + 2
		url_2 = url_post[url_2_slash:]

		ex_slash = url_2[1:].find("/")
		url_add = ""

		if (ex_slash != -1):
			url_add = "_" + url_2[1:]
			add_cut = url_add.find("/")
			url_add = url_add[:add_cut]

		dir1 = url_post[:url_1_slash]
		dir2 = url_post[url_1_slash + 1:url_2_slash] + url_add

		local_path = "img/" + dir1 + "/" + dir2
		new_path_stem = "img/all/"
		new_path = new_path_stem + img_tag
		#print(local_path)

		Path(local_path).mkdir(parents=True, exist_ok=True)
		Path(new_path_stem).mkdir(parents=True, exist_ok=True)
		#print(local_path + "/" + img_tag)

		#Image Sorter
		total_path = local_path + "/" + img_tag
		cut_path = total_path
		if path.exists(total_path) and path.exists(new_path_stem):
			shutil.copy(total_path, new_path_stem)
			row["RAW_URL"] = dl_url
			row["SPLIT_PATH"] = total_path
			row["PATH"] = new_path
			row["ARTIST_RAW"] = dir2
			row["INITIAL"] = dir1
			print(row)
			csv_writer.writerow(row)
			# #img = Image.open(total_path)
			# img = 0
			# curr_dict = {}
			# curr_dict["path"] = total_path
			# curr_dict["artist_raw"] = dir2
			# curr_dict["initial"] = dir1
			# curr_dict["artist"] = row["AUTHOR"]
			# curr_dict["life"] = row["BORN-DIED"]
			# curr_dict["img_tag"] = img_tag
			# curr_dict["date"] = row["DATE"]
			# curr_dict["tech"] = row["TECHNIQUE"]
			# curr_dict["loc"] = row["LOCATION"]
			# curr_dict["img_url"] = dl_url
			# curr_dict["form"] = row["FORM"]
			# curr_dict["type"] = row["TYPE"]
			# curr_dict["school"] = row["SCHOOL"]
			# curr_dict["period"] = row["TIMELINE"]
			# curr_dict["img_obj"] = img
			# img_dict[row["TITLE"]] = curr_dict
			# #img.close()
		elif "zzzarchi" in total_path:
			print("Skipping: " + total_path)
		else:
			print("Path: " + total_path + " does not exist!")


#img_data = pd.DataFrame.from_dict(img_dict)
#print(img_data)