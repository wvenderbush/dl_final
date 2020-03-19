import csv
import requests
import shutil
from pathlib import Path


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
		if (line_count >= 10):
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

		dl_url = url_prehtml + url_posthtml
		#print(dl_url)

		#File Path Generator
		url_post = url_posthtml[1:]
		url_1_slash = url_post.find("/")
		url_1 = url_post[url_1_slash:]
		url_2_slash = url_1[1:].find("/") + 2

		dir1 = url_post[:url_1_slash]
		dir2 = url_post[url_1_slash + 1:url_2_slash]

		Path("img/" + dir1 + "/" + dir2 ).mkdir(parents=True, exist_ok=True)

		#Image Downloader

		image_url = dl_url
		resp = requests.get(image_url, stream=True)
		
		local_url = "img/" + url_posthtml[1:]
		print("Downloading: " + local_url)
		#print(local_url)
		local_file = open(local_url, 'wb+')
		resp.raw.decode_content = True
		shutil.copyfileobj(resp.raw, local_file)
		del resp

		line_count += 1



		# if line_count == 0:
		# 	print(f'Column names are {", ".join(row)}')
		# 	line_count += 1
		# #print(f'\t{row["name"]} works in the {row["department"]} department, and was born in {row["birthday month"]}.')
		

	#print(f'Processed {line_count} lines.')


