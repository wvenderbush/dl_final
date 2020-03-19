import csv
import requests
import shutil
from pathlib import Path
from os import path


# with open('catalog.csv') as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     line_count = 0
#     for row in csv_reader:
#         if line_count == 0:
#             print(f'Column names are {", ".join(row)}')
#             line_count += 1
#         else:
#             print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
#             line_count += 1
#     print(f'Processed {line_count} lines.')

with open('catalog.csv', mode='r', encoding="latin-1") as csv_file:
	csv_reader = csv.DictReader(csv_file)
	line_count = 0
	for row in csv_reader:
		line_count += 1
		if (line_count > 75):
			break

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
		#print(local_path)

		Path(local_path).mkdir(parents=True, exist_ok=True)
		print(local_path + "/" + img_tag)

		#Image Downloader
		if path.exists("img/" + local_path + "/" + img_tag):
			print("Already Exists: " + local_path + "/" + img_tag)
		else:
			image_url = dl_url
			print(image_url)
			resp = requests.get(image_url, stream=True)
			
			local_url = "img/" + local_path
			print(str(line_count) + " - Downloading: " + local_url)
			#print(local_url)
			local_file = open(local_url, 'wb+')
			resp.raw.decode_content = True
			shutil.copyfileobj(resp.raw, local_file)
			del resp

		



		# if line_count == 0:
		# 	print(f'Column names are {", ".join(row)}')
		# 	line_count += 1
		# #print(f'\t{row["name"]} works in the {row["department"]} department, and was born in {row["birthday month"]}.')
		

	#print(f'Processed {line_count} lines.')


