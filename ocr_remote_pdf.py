import urllib.request
import numpy as np
import sys
import cv2
import os
from PIL import Image
import pytesseract
import cv2
from bs4 import BeautifulSoup
import statistics
import math
import os 
import json
import re
import datetime


# global document information
global_doc = {}

keyword_value_group = {
	'bl no':"Shipment - Air Waybill Number",
	'lc no':"LC - Number",  
	'commodity':"",
	'origin':"",
	'quantity':"",
	'invoice no':"Invoice - Number",
	'invoice date':"Invoice - Date",
	'to':"Importer - Name",
	'port of discharge':"Shipment - Port - Destination",
	'from':"Exporter - Name",
	'unit price':"",
	'amount':"",
	'quality':"",
	'trade terms':"Shipment - Term",
	'contract no': "Contract - Number"		
}


pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"

def image_smoothening(img):
	BINARY_THREHOLD = 180
	ret1, th1 = cv2.threshold(img, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
	ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	blur = cv2.GaussianBlur(th2, (1, 1), 0)
	ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	return th3


def remove_noise_and_smooth(img):
	filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
	kernel = np.ones((1, 1), np.uint8)
	opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
	closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
	img = image_smoothening(img)
	or_image = cv2.bitwise_or(img, closing)
	return or_image

def stk_ofl_smooth_img(im):

	# smooth the image with alternative closing and opening
	# with an enlarging kernel
	morph = im.copy()

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
	morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
	morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

	# take morphological gradient
	gradient_image = cv2.morphologyEx(morph, cv2.MORPH_GRADIENT, kernel)

	# split the gradient image into channels
	image_channels = gradient_image

	channel_height, channel_width = image_channels.shape

	# apply Otsu threshold to each channel
	
	_, image_channels = cv2.threshold(~image_channels, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
	image_channels = np.reshape(image_channels, newshape=(channel_height, channel_width, 1))


	# save the denoised image
	return image_channels


def delete_line(input_image_path, output_image_path, is_detect_phrase=False):
	image_name = input_image_path.split('/')[-1].split('.')[0]

	# Load the image
	src = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
	global_doc['origin_img'] = src

	# Check if image is loaded fine
	if src is None:
		print ('Error opening image: ' + input_image_path)
		return -1

	# Transform source image to gray if it is not already
	if len(src.shape) != 2:
		gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
	else:
		gray = src
	gray = remove_noise_and_smooth(gray)

	# Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
	gray = cv2.bitwise_not(gray)
	bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
								cv2.THRESH_BINARY, 15, -2)

	bw_dialte = cv2.dilate(bw, np.ones((1,1)))
	#bw_dialte = cv2.erode(bw_dialte, np.ones((2,2)))

	# Create the images that will use to extract the horizontal and vertical lines
	horizontal = np.copy(bw_dialte)
	vertical = np.copy(bw_dialte)
	# [init]

	# [horiz]
	# Specify size on horizontal axis
	cols = horizontal.shape[1]
	horizontal_size = 200

	# Create structure element for extracting horizontal lines through morphology operations
	horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

	# Apply morphology operations
	horizontal = cv2.dilate(horizontal, np.ones((4, 4)))
	horizontal = cv2.erode(horizontal, horizontalStructure)
	horizontal = cv2.dilate(horizontal, horizontalStructure)
	

	horizontal = cv2.bitwise_not(horizontal)
	#cv2.imwrite('sample_result_line_result/delete_line_{}_horiz.jpg'.format(image_name), horizontal)
	#global_doc['horizontal_img'] = horizontal


	rows = vertical.shape[0]
	verticalsize = 100

	# Create structure element for extracting vertical lines through morphology operations
	verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))

	# Apply morphology operations
	vertical = cv2.dilate(vertical, np.ones((4, 4)))
	vertical = cv2.erode(vertical, verticalStructure)
	vertical = cv2.dilate(vertical, verticalStructure)
	
	vertical = cv2.bitwise_not(vertical)
	# show_wait_destroy("vertical_bit", vertical)
	#cv2.imwrite('sample_result_line_result/delete_line_{}_vert.jpg'.format(image_name), vertical)
	#global_doc['vertical_img'] = vertical

	bw = cv2.bitwise_and(bw, bw, mask=horizontal)
	masked_img = cv2.bitwise_and(bw, bw, mask=vertical)
	masked_img = cv2.bitwise_not(masked_img)

	cv2.imwrite(output_image_path, masked_img)
	# [smooth]

	#table_img = cv2.bitwise_or(mask)
	if is_detect_phrase:
		get_table_pixel(horizontal, vertical)

	return 0


def get_erode(image):
	bitwise_img = cv2.bitwise_not(image)
	bitwise_img = cv2.erode(bitwise_img, np.ones((4,4)))
	result = cv2.bitwise_not(bitwise_img)
	return result


def get_table_pixel(horizontal, vertical):

	verti_black = get_black_pixel(get_erode(vertical))
	table_black = set(verti_black) 
	horiz_black = get_black_pixel(get_erode(horizontal))
	table_black = set(horiz_black) 

	global_doc['table_black'] = set(horiz_black) | set(verti_black)


def get_black_pixel(image):
	result = []

	height, width = image.shape
	print('SHAPE: ', width, height)
	check = []

	for x in range(width):
	
	
		# print(image[x])
		check = []
		for y in range(height):	

			if image[y,x] == 0:
				result.append((x,y))
				check.append((x,y))
		# if(len(check) > 0):
		# 	print('\nCHECK-LIST:{}:\n'.format(x), check)

	return result


def filter_cell(width, height):
	if width < 3 and height > 5:
		return False

	if width < 5 and height > 100:
		return False

	if height > 100 and width < 5:
		return False

	return True


def filter_cell_ref(text, width, height, char_height):
	if len(text.strip()) == 0:
		if height < char_height*0.2 or width < char_height*0.2:
			return False


	if height < char_height*0.5 and width > char_height*3:
		return False

	if height > char_height*3 and width < char_height*0.5:
		return False

	# return True
	# if height < char_height + 5 and height > char_height - 5:
	# 	return True
	# return False 
	return True


def filter_text(s):
	return (''.join(filter(str.isalpha, s))).strip() 

def filter_value(s):
	return re.sub(r'\W+', '', s)


def write_file(data, file_path):
	file = open(file_path,'w') 
	file.write(data) 
	file.close() 


def one_line(points_first, points_next):
	if (abs(points_next[0] - points_first[2]) < global_doc['median_height'] 
		and abs(points_next[1] - points_first[1]) < global_doc['median_height']):
		return True

	return False

def detect_value_accumulate_cord(new_cor, old_cor):
	if old_cor == [0,0,0,0]:
		return new_cor
	else:
		old_cor[1] = min(old_cor[1],new_cor[1])
		old_cor[3] = max(old_cor[3],new_cor[3])
		old_cor[2] = new_cor[2]
	
	return old_cor


def detect_value(i, j, keyword, word_box_list, value_length):
	'''
	Get the keyword-value
	i - the start of keyword
	j - the end of keyword 
	word_box_list - list of word-box data 
	value_length - number of word in the value
	'''
	result = ''
	
	print(keyword + ' ---------------------------')
	old_cor = [0,0,0,0]
	if value_length != -1:
		count = 0
		t = 0
		while count < value_length:
			t += 1
			text = word_box_list[i+j+t][1]
			filtered = filter_value(text)
			if len(filtered) > 0:
				result += text
				old_cor = detect_value_accumulate_cord(word_box_list[i+j+t][0], old_cor)
				count += 1
				if count < value_length:
					result += ' '

	else:
		count = 0
		t = 0
		while True:
			points_first = word_box_list[i+j+t][0]
			t += 1
			points_next = word_box_list[i+j+t][0]
			
			if one_line(points_first, points_next):
				text = word_box_list[i+j+t][1]
				result += text + ' '
				old_cor = detect_value_accumulate_cord(word_box_list[i+j+t][0], old_cor)
				count += 1
			else:
				break	

	#print(keyword + ' ---------------------------')
	return result, old_cor


def detect_box(input_url, paths, output_folder, keywords, keyword_value_length, is_detect_phrase=False):
	raw_result = []
	for idx, path in enumerate(paths):
		raw = {}
		raw['raw_id'] = str(idx) + '_' + path
		raw['url'] = input_url
		raw['annotion'] = []

		img = cv2.imread(path)
		hocr = pytesseract.image_to_pdf_or_hocr(img, extension='hocr', config='--psm 1')
		html = ( hocr.decode("utf-8") )
		#print(html)
		write_file(html, output_folder + str(idx) + '_' + path + '.txt')



		soup = BeautifulSoup(html, 'html.parser')

		lines = soup.find_all('span', class_='ocrx_word')

		# print(soup.text)

		base_name = os.path.basename(path.replace('.jpg', ''))
		result_prefix = output_folder + 'tmp_' +  base_name

		count = 0

		height, width, channels = img.shape

		#doc_height
		#doc_width

		word_box_list = []
		width_box_list = []
		height_box_list = []


		# Detect word statistics: median height
		for line in lines:
			count += 1
			# print(line.text)
			box = line.attrs['title'].split(';')[0]
			#print(box)
			box = box.split()[1:]
			points = [int(i) for i in box]
			# print(line.text, ' ### ', points)
			#draw.rectangle(((points[0], points[1]), (points[2], points[3])), outline='red')
			
			width_box = math.fabs(points[2] - points[0])
			height_box = math.fabs(points[3] - points[1])
			width_box_list.append(width_box)
			height_box_list.append(height_box)
			#print(width_box, height_box)
			#print(points)
			# if filter_cell(width_box, height_box):
			# 	word_box_list.append((points, line.text))
		print('\n\n\n')

		median_height = statistics.median(height_box_list)
		print('--->> character height: ' + str(median_height))
		global_doc['median_height'] = median_height

		# Detect word location and text
		for line in lines:
			
			#print(line.text)
			box = line.attrs['title'].split(';')[0]
			#print(box)
			box = box.split()[1:]
			points = [int(i) for i in box]
			#draw.rectangle(((points[0], points[1]), (points[2], points[3])), outline='red')
			
			width_box = math.fabs(points[2] - points[0])
			height_box = math.fabs(points[3] - points[1])

			if filter_cell_ref(line.text, width_box, height_box, median_height):
				count += 1
				word_box_list.append((points, line.text))
				#print('-->>: ', points, line.text)
				cv2.rectangle(img,(points[0], points[1]), (points[2], points[3]),(0,123,60),1)

		# Detect keyword and value
		for i in range(len(word_box_list)):
			points_i = []
			text_i = []
			for j in range(3):
				if i + j < len(word_box_list):
					points_j, text_j = word_box_list[i+j]
					text_j = filter_text(text_j.lower())
					points_i.append(points_j)
					text_i.append(text_j)

					word_j = ' '.join(text_i)

					if word_j in keywords:
						value_text, cor = detect_value(i,j, word_j, word_box_list, keyword_value_length[word_j])
						#word_box_list[i+j+1][1]
						print('+++>>>:', word_j, points_i, value_text)
						# for t in range(len(points_i)):
						# 	points = points_i[t]
						# 	cv2.rectangle(img,(points[0]-2, points[1]-2), (points[2]+2, points[3]+2),(0,255,0),2)
						annotion = {}
						annotion['doc_width'] = width
						annotion['doc_height'] = height
						annotion['anchor_x'] = cor[0]
						annotion['anchor_y'] = cor[1]
						annotion['anchor_width'] = cor[2]-cor[0]
						annotion['anchor_height'] = cor[3]-cor[1]
						
						annotion['id'] = str(i) + '_' + str(j) + '_' + word_j
						annotion['status'] = 'active'
						annotion['text_by_ocr'] = word_j
						annotion['text_by_editor'] = ''
						annotion['anno_type'] = keyword_value_group[word_j]
						annotion['key'] = word_j
						annotion['value'] = value_text
						annotion['amended_from_anno_id'] = ''
						annotion['author_type'] = 0

						raw['annotion'].append(annotion)

						cv2.rectangle(img,
							(cor[0]-2, cor[1]-2), 
							(cor[2]+2, cor[3]+2),
							(0,255,0),2)


		# Draw word next-neighbor	
		# for i in range(len(word_box_list)-1):
		# 	points_1, _ = word_box_list[i]
		# 	points_2, _ = word_box_list[i+1]
		# 	x1 = points_1[2]
		# 	y1 = (points_1[1] + points_1[3])/2
		# 	x2 = points_2[0]
		# 	y2 = (points_2[1] + points_2[3])/2
		# 	# cv2.line(img, (int(x1), int(y1)), ( int(x2), int(y2)), (255, 0, 0), 2)

		# Detect word-phrase
		if is_detect_phrase:
			word_phrase = detect_word_phrase(word_box_list, img)


			find_horiz_distance(word_phrase, img)

			combine_phrase = combine_horiz_phrase(word_phrase, img)
			combine_phrase = filter_phrase(combine_phrase)

			next_horiz_phrase(combine_phrase, img)

			next_verti_phrase(combine_phrase, img)

			draw_phrase_neighbor(combine_phrase, img)

			# print('--->>>> IMAGE-TYPE:: ', type(img), type(global_doc['table_black']))

			print('--->>> BITWISE 2 images')
			#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			#img = cv2.bitwise_and(gray, global_doc['vertical_img'])
			#cv2.addWeighted(img,0.5,global_doc['vertical_img'],0.5,0)

		output_with_box = result_prefix + '.jpg'
		#img.save(output_with_box, 'PNG')
		cv2.imwrite(output_with_box, img)

		raw_result.append(raw)

	return raw_result



def find_horiz_distance(word_phrase, img):
	return []


def check_phrase(phrase):
	if (abs(phrase[2][0]-phrase[2][2]) > global_doc['median_height']/2 
			and abs(phrase[2][1]-phrase[2][3]) > global_doc['median_height']/2):
		return True
	return False

def filter_phrase(combine_phrase):
	result = []

	for phrase in combine_phrase:
		if (abs(phrase[2][0]-phrase[2][2]) > global_doc['median_height']/2 
			and abs(phrase[2][1]-phrase[2][3]) > global_doc['median_height']/2):
			result.append(phrase)



	return result


def line_points(img, p1, p2):
	x0 = p1[0]
	y0 = p1[1]
	x1 = p2[0]
	y1 = p2[1]
	deltax = x1-x0
	if deltax != 0:
		dxsign = int(abs(deltax)/deltax)
	else:
		dxsign = 0
	deltay = y1-y0
	if deltay != 0:
		dysign = int(abs(deltay)/deltay)
	else:
		dysign = 0
	deltaerr = abs(deltay/deltax)
	error = 0
	y = y0
	for x in range(x0, x1, dxsign):
		yield x, y
		error = error + deltaerr
		while error >= 0.5:
			y += dysign
			error -= 1
	yield x1, y1


def check_cross_table(points_1, points_2, img):
	x1 = points_1[2]
	y1 = (points_1[1] + points_1[3])/2
	x2 = points_2[0]
	y2 = (points_2[1] + points_2[3])/2

	points = list(line_points(img, (int(x1), int(y1)), ( int(x2), int(y2)) ))
	#print('points: \n', points, '\n')
	line_set = set(points)
	intersection = line_set & global_doc['table_black']
	#print('intersection: \n', intersection, '\n')
	if len(intersection) > 0:
		return False

	return True

def draw_phrase_neighbor(word_phrase, img):
	for idx, phrase in enumerate(word_phrase):
		
		if idx in global_doc['horiz_neighbor']:	
			horiz_next_id = global_doc['horiz_neighbor'][idx] 
			next_phrase = word_phrase[horiz_next_id]
			draw_next_neighbor(
				phrase[2][2], (phrase[2][1] + phrase[2][3]) / 2, 
				next_phrase[2][0], (next_phrase[2][1] + next_phrase[2][3])/2, 
				img)
		
		if idx in global_doc['verti_neighbor']:	
			verti_next_id = global_doc['verti_neighbor'][idx]
			next_phrase = word_phrase[verti_next_id]
			draw_next_neighbor(
				(phrase[2][0] + phrase[2][2]) / 2, phrase[2][3], 
				(next_phrase[2][0] + next_phrase[2][2]) / 2, next_phrase[2][1], 
				img)



def draw_next_neighbor(x1, y1, x2, y2, img):
	# x1 = points_1[2]
	# y1 = (points_1[1] + points_1[3])/2
	# x2 = points_2[0]
	# y2 = (points_2[1] + points_2[3])/2
	cv2.line(img, (int(x1), int(y1)), ( int(x2), int(y2)), (255, 0, 0), 2)


def next_horiz_phrase(word_phrase, img):
	result = []
	global_doc['horiz_neighbor'] = {}
	global_doc['horiz_neighbor'] = next_neighbor_phrase(word_phrase, img, next_horiz_one_line)
	print('Horizontal ---------------', global_doc['horiz_neighbor'])



def next_verti_phrase(word_phrase, img):
	result = []
	global_doc['verti_neighbor'] = {}
	global_doc['verti_neighbor'] = next_neighbor_phrase(word_phrase, img, next_verti_one_line)
	print('Vertical -----------------', global_doc['verti_neighbor'])





def next_neighbor_phrase(word_phrase, img, next_one_line):
	result = {}
	print('-----------------------')
	for idx, phrase in enumerate(word_phrase):
		# if idx not in remove_idx:
		# 1. Find 1 closest phrase to the left but consider horizontal-one-line
		neighbor_next_phrase = find_neighbor_combine(idx, phrase, word_phrase, next_one_line)
		# 2. Find 1 closest phrase to the down but consider vertical-one-line
		# verti_next_phrase = find_verti_next(idx, phrase, word_phrase)
		next_idx = neighbor_next_phrase[0]
		next_phrase = neighbor_next_phrase[1]
		if next_idx != -1:	
			# remove_idx.append(next_idx)
			# print(phrase)
			# print(next_phrase)
			if check_cross_table(phrase[2], next_phrase[2], img):	
				# draw_next_neighbor(phrase[2], next_phrase[2], img)
				result[idx] = next_idx

			#print(phrase[1], ' ::: ', next_phrase[1])
			#print('\n-------------\n')
	return result


def combine_horiz_phrase(word_phrase, img):
	result = []
	next_idx_map = {}

	for idx, phrase in enumerate(word_phrase):
		# if idx not in remove_idx:
		# 1. Find 1 closest phrase to the left but consider horizontal-one-line
		horiz_next_phrase = find_neighbor_combine(idx, phrase, word_phrase, horiz_one_line)
		# 2. Find 1 closest phrase to the down but consider vertical-one-line
		# verti_next_phrase = find_verti_next(idx, phrase, word_phrase)
		next_idx = horiz_next_phrase[0]
		next_phrase = horiz_next_phrase[1]
		# remove_idx.append(next_idx)
		next_idx_map[idx] = next_idx

	checked_idx = set()
	# print('Next idx map: ', next_idx_map)
	for idx, phrase in enumerate(word_phrase):
		if idx not in checked_idx:
			current_idx = idx
			checked_idx.add(current_idx)

			while next_idx_map[current_idx] != -1:
				next_idx = next_idx_map[current_idx]
				next_phrase = word_phrase[next_idx]


				phrase[0].extend(next_phrase[0])
				phrase[1] += ' ' + next_phrase[1]

				phrase[2][2] = next_phrase[2][2]
				phrase[2][1] = min(phrase[2][1], next_phrase[2][1])
				phrase[2][3] = max(phrase[2][3], next_phrase[2][3])

				current_idx = next_idx
				checked_idx.add(current_idx)

			if check_phrase(phrase):
				result.append(phrase)
				#print('phrase: {}'.format(phrase))

				cv2.rectangle(img,(phrase[2][0], phrase[2][1]), (phrase[2][2], phrase[2][3]),(0,0,255),1)

	return result


def find_neighbor_combine(idx, phrase, word_phrase, one_line_func):
	result = (-1, None)
	closest_dist = float('inf')
	for can_idx, candidate in enumerate(word_phrase):
		if can_idx != idx:
			if one_line_func(phrase[2], candidate[2]):
				center_1 = ( (phrase[2][0] ), (phrase[2][1] + phrase[2][3]) / 2)
				center_2 = ( (candidate[2][0] ), (candidate[2][1] + candidate[2][3]) / 2)
				dist = distance(center_1, center_2)
				if closest_dist > dist:
					closest_dist = dist
					result = (can_idx, candidate)

	return result


def horiz_one_line(points_first, points_next):
	if (
		abs( (points_next[3]-points_next[1]) - (points_first[3]-points_first[1]) ) < global_doc['median_height']/2 and
		abs(points_next[0] - points_first[2]) < global_doc['median_height'] and
		abs(points_next[1] - points_first[1]) < global_doc['median_height']/4):
		return True
	return False


def next_horiz_one_line(points_first, points_next):
	if (
		abs( (points_next[3]-points_next[1]) - (points_first[3]-points_first[1]) ) < global_doc['median_height']/2 and
		points_first[2] < points_next[0] and
		abs(points_next[1] - points_first[1]) < global_doc['median_height']/4):
		return True
	return False

def next_verti_one_line(points_first, points_next):
	if (points_first[0] < points_next[2] and points_first[2] > points_next[0] and 
		points_first[3] < points_next[1] and
		abs(points_next[1] - points_first[3]) < global_doc['median_height']):
		return True
	return False


def detect_word_phrase(word_box_list, img):
	result = []

	# phrase (word_list [word], text, points)
	start = 0

	while start < len(word_box_list):
		t = 0
		count = 0
		word_list = [word_box_list[start]]
		text = word_box_list[start][1]
		points = word_box_list[start][0].copy()
		while start + t + 1 < len(word_box_list):
			points_first = word_box_list[start+t][0]
			t += 1
			points_next = word_box_list[start+t][0]
			
			if one_line(points_first, points_next):
				text += ' ' + word_box_list[start+t][1] 
				word_list.append(word_box_list[start+t])
				count += 1
				points[2] = points_next[2]
				points[1] = min(points[1], points_next[1])
				points[3] = max(points[3], points_next[3])
			else:
				break
		start = start + count + 1
		result.append([word_list, text, points])
		# print('phrase: {} {} {}'.format(word_list, text, points))
		# cv2.rectangle(img,(points[0], points[1]), (points[2], points[3]),(0,0,255),1)


	return result



# helper function
        
def distance(a, b):
	"""
	Squared distance between points a & b
	"""
	return (a[0]-b[0])**2 + (a[1]-b[1])**2



# main function

def process_local_img_file(image_paths, output_folder, keywords, keyword_value_length):

	d_image_paths = []
	for p_origin in image_paths:
		p = p_origin.split('/')[-1]
		output_p = p.replace('.jpg', '') + '_del_line.jpg'
		delete_line(p_origin, output_p)
		d_image_paths.append(output_p)
		# os.remove(p)


	if not os.path.exists(output_folder):
	    os.makedirs(output_folder)

	print(d_image_paths)
	raw = detect_box('input_url', d_image_paths, output_folder, keywords, keyword_value_length)


	# for p in d_image_paths:
	# 	os.remove(p)

	return raw



def process_remote_pdf_file(input_url, local_file, output_folder, keywords, keyword_value_length):

	urllib.request.urlretrieve(input_url, local_file)


	from pdf2image import convert_from_path
	pages = convert_from_path(local_file)

	image_paths = []
	for idx, page in enumerate(pages):
		file_img = local_file.replace('.pdf', '') + '_' + str(idx) + '.jpg'
		page.save(file_img, 'JPEG')
		image_paths.append(file_img)


	os.remove(local_file)


	d_image_paths = []
	for p in image_paths:

		output_p = p.replace('.jpg', '') + '_del_line.jpg'
		delete_line(p, output_p)
		d_image_paths.append(output_p)
		os.remove(p)


	if not os.path.exists(output_folder):
	    os.makedirs(output_folder)

	raw = detect_box(input_url, d_image_paths, output_folder, keywords, keyword_value_length)


	for p in d_image_paths:
		os.remove(p)

	return raw


def process_local_image():
	#input_url = 'http://35.224.102.42/files/lc_document/05 - Packing list.pdf'
	#input_url = "http://35.224.102.42/files/lc_document/08 - Shipment Receiptp.pdf"
	#'http://35.224.102.42/files/lc_document/10 - Bill of Exchange.pdf'
	#"http://35.224.102.42/files/lc_document/01%20-%20Certificate%20of%20Quality.pdf"
	keywords = set([
		'bl no',
		'lc no',  
		'commodity',
		'origin',
		'quantity',
		'invoice no',
		'invoice date',
		'to',
		'port of discharge',
		'from',
		'quantity',
		'unit price',
		'amount',
		'commodity',
		'quality',
		'trade terms',
		'contract no'
		])
	keyword_value_length = {
		'bl no':1,
		'lc no':1,  
		'commodity':-1,
		'origin':-1,
		'quantity':-1,
		'invoice no':1,
		'invoice date':3,
		'to':-1,
		'port of discharge':-1,
		'from':-1,
		'unit price':-1,
		'amount':-1,
		'quality':-1,
		'trade terms':-1,
		'contract no':1		
	}

	keyword_value_group = {
		'bl no':"Shipment - Air Waybill Number",
		'lc no':"LC - Number",  
		'commodity':"",
		'origin':"",
		'quantity':"",
		'invoice no':"Invoice - Number",
		'invoice date':"Invoice - Date",
		'to':"Importer - Name",
		'port of discharge':"Shipment - Port - Destination",
		'from':"Exporter - Name",
		'unit price':"",
		'amount':"",
		'quality':"",
		'trade terms':"Shipment - Term",
		'contract no': "Contract - Number"		
	}


	image_paths = ['python/data/1.jpg']
	output_folder = 'python/result/sample/'
	raw = process_local_img_file(image_paths, output_folder, keywords, keyword_value_length)
	# print(raw)
	return raw



def process_url_image(input_url):
	#input_url = 'http://35.224.102.42/files/lc_document/05 - Packing list.pdf'
	#input_url = "http://35.224.102.42/files/lc_document/08 - Shipment Receiptp.pdf"
	#'http://35.224.102.42/files/lc_document/10 - Bill of Exchange.pdf'
	#"http://35.224.102.42/files/lc_document/01%20-%20Certificate%20of%20Quality.pdf"
	keywords = set([
		'bl no',
		'lc no',  
		'commodity',
		'origin',
		'quantity',
		'invoice no',
		'invoice date',
		'to',
		'port of discharge',
		'from',
		'quantity',
		'unit price',
		'amount',
		'commodity',
		'quality',
		'trade terms',
		'contract no'
		])
	keyword_value_length = {
		'bl no':1,
		'lc no':1,  
		'commodity':-1,
		'origin':-1,
		'quantity':-1,
		'invoice no':1,
		'invoice date':3,
		'to':-1,
		'port of discharge':-1,
		'from':-1,
		'unit price':-1,
		'amount':-1,
		'quality':-1,
		'trade terms':-1,
		'contract no':1		
	}
	now = datetime.datetime.now().strftime("%I_%M_%p_%B_%d_%Y")
	file_name = input_url.split('/')[-1]
	local_file = 'tmp_' + str(now) + '_' + file_name
	print(local_file)
	#"sample_file_line_{}.pdf".format(index)
	output_folder = 'result/sample/'
	raw = process_remote_pdf_file(input_url, local_file, output_folder, keywords, keyword_value_length)
	print('\n\nResult:\n', raw)
	return raw



def read_json_and_process():
	json_raw = '{"message":{"total_file":5,"file_url":[{"file_url":"/files/10_-_Bill_of_Exchange.pdf","type":"BE"}],"doc_id":"LCV-1901200003","related_path":"http://35.224.102.42:8080"}}' 
	#'{"message":{"total_file":5,"file_url":[{"file_url":"/files/10_-_Bill_of_Exchange.pdf","type":"BE"}],"doc_id":"LCV-1901200003","related_path":"http://35.224.102.42:8080"}}'
	json_data = json.loads(json_raw)
	#read_json()
	message = json_data['message']
	total_file = message['total_file']
	file_url = message['file_url']
	doc_id = message['doc_id']
	related_path = message['related_path']

	result = {}
	result['doc_id'] = doc_id
	result['title'] = ''
	result['created_time'] = ''
	result['page'] = []

	for index, url in enumerate(file_url):
		file_path = url['file_url']
		file_type = url['type']
		page_result = {}
		input_url = related_path + file_path
		index = index + 1
		page_result['page_id'] = input_url

		raw = process_url_image(input_url, index)
		page_result['raw'] = raw 

		result['page'].append(page_result)

	# print(result)
	return result


def read_json():
	import requests
	headers = {
		'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:64.0) Gecko/20100101 Firefox/64.0',
		'Accept': 'application/json',
		'Referer': 'http://35.224.102.42:8080/desk',
		'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
		'X-Requested-With': 'XMLHttpRequest'
	}
	body = 'cmd=erpnext.projects.doctype.lc_verfication.lc_verfication.generate_document_sample'
	url = 'http://35.224.102.42:8080/'
	response = requests.post(
        url,
        body, 
        headers=headers
        )
	
	r = json.loads(response.text)
	return r


# @Todo: build line-phrase-list -->> Draw phrase-box
# Phrase - similar to word -->> text + box 
# Phrase-box: first-word points[0,1] + last-word points[2,3]
# @Todo:

if __name__ == '__main__':
	#process_url_image('http://orekahq.com:8001/files/05_-_Packing_list.pdf')
	#read_json_and_process()
	process_local_image()
	#print(read_json())