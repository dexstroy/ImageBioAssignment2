import cv2

cascadeRightEar = cv2.CascadeClassifier("haarcascade_mcs_rightear.xml")
cascadeLeftEar = cv2.CascadeClassifier("haarcascade_mcs_leftear.xml")

filepath = "./test/"
savePath = "./testDetected/"


def get_boxes_min_max(image_name):
	slika = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)

	_, threshold = cv2.threshold(slika, 110, 255, cv2.THRESH_BINARY)

	contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	boxes = []

	for box in contours:
		x_min = 200000
		x_max = 0
		y_min = 200000
		y_max = 0
		for edge in box:
			x = edge[0][0]
			y = edge[0][1]
			if x < x_min:
				x_min = x
			if x > x_max:
				x_max = x
			if y < y_min:
				y_min = y
			if y > y_max:
				y_max = y
		boxes.append((x_min, x_max, y_min, y_max))

	return boxes


def get_detected_boxes_min_max(detection_list):
	boxes = []
	for x, y, w, h in detection_list:
		x_min = x
		x_max = x + w
		y_min = y
		y_max = y + h
		boxes.append((x_min, x_max, y_min, y_max))
	return boxes


def detect_right_ear(slika):
	detection_list = cascadeRightEar.detectMultiScale(slika, 1.03, 3)
	return detection_list


def detect_left_ear(slika):
	detection_list = cascadeLeftEar.detectMultiScale(slika, 1.03, 3)
	return detection_list


def visualisation(slika, detection_list_right, detection_list_left):
	for x, y, w, h in detection_list_right:
		cv2.rectangle(slika, (x, y), (x + w, y + h), (128, 255, 0), 4)

	for x, y, w, h in detection_list_left:
		cv2.rectangle(slika, (x, y), (x + w, y + h), (128, 255, 0), 4)

	cv2.imwrite(savePath + filename, slika)


def get_intersection_ratio(rectA, rectB):
	#struktura: (x_min, x_max, y_min, y_max)
	'''
	dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
	dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
	'''

	dx = min(rectA[1], rectB[1]) - max(rectA[0], rectB[0])
	dy = min(rectA[3], rectB[3]) - max(rectA[2], rectB[2])

	if (dx >= 0) and (dy >= 0):
		intersect_area = dx * dy
		areaRectA = (rectA[1] - rectA[0]) * (rectA[3] - rectA[2])
		areaRectB = (rectB[1] - rectB[0]) * (rectB[3] - rectB[2])

		return intersect_area / (areaRectA + areaRectB - intersect_area)

	return 0
ratios = []
for stev in range(1, 21):
	filename = "0" * (4 - len(str(stev))) + str(stev) + ".png"



	img = cv2.imread(filepath + filename)
	detectionListRight = detect_right_ear(img)
	detectionListLeft = detect_left_ear(img)

	detected_boxes = get_detected_boxes_min_max(detectionListLeft) + get_detected_boxes_min_max(detectionListRight)
	marked_boxes = get_boxes_min_max("./testannot_rect/" + filename)

	if len(detected_boxes) != 0:
		print(filename)

	for detected_box in detected_boxes:
		maxRatio = 0
		for marked_box in marked_boxes:
			print("Ratio:")
			ratio = get_intersection_ratio(detected_box, marked_box)
			if ratio > maxRatio:
				maxRatio = ratio
		if maxRatio != 0:
			ratios.append(maxRatio)

	visualisation(img, detectionListRight, detectionListLeft)


print("AverageIoURatio:")
print(sum(ratios) / len(ratios))
