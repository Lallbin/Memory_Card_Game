import numpy as np
import cv2
import os
from glob import glob

def class_to_name(class_number):
    
    class_number = int(class_number)
    
    names = ["back", "parrot", "turtle", "small dog", "medium dog", "large dog", "hamster", "pig", "cat"]
    
    name = names[class_number]
    
    return name


def extract_sift_features(image):

    if isinstance(image, str):
        gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if gray is None:
        print(f'Error: Could not load image {gray}')
        return None, None
    
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors
    

def match_descriptors(descriptors, training_data):

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(descriptors.astype(np.float32), training_data['descriptors'].astype(np.float32), k=2)

    ratio_thresh = 0.8
    good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]

    return good_matches


def create_training_data(training_folder):

    descriptors_list = []
    labels_list = []
    class_names = sorted([d for d in os.listdir(training_folder) if os.path.isdir(os.path.join(training_folder, d))])

    for label, class_name in enumerate(class_names):
        image_paths = glob(os.path.join(training_folder, class_name, '*.jpg'))
        for image_path in image_paths:
            _, descriptors = extract_sift_features(image_path)
            if descriptors is not None:
                descriptors_list.append(descriptors)
                labels_list.extend([label] * descriptors.shape[0])
    
    descriptors_array = np.vstack(descriptors_list) if descriptors_list else np.array([])
    labels_array = np.array(labels_list, dtype=np.int32)
    
    training_data = {
        'descriptors': descriptors_array,
        'labels': labels_array,
        'names': class_names
    }

    return training_data


def classify_card(image, training_data):
    
    labels = training_data["labels"]
    names = training_data["names"]

    sift_features = extract_sift_features(image)

    try:
        matches = match_descriptors(sift_features[1], training_data)
    except:
        return 0
    
    votes = np.zeros(len(names))
    
    for match in matches:
        vote = labels[match.trainIdx]
        votes[vote] += 1
    
    label = np.argmax(votes)
    
    return label


def resize_image(image, scaling_factor):
    w, h = image.shape[:2]
    new_w = int(w*scaling_factor)
    new_h = int(h*scaling_factor)
    
    resized_image = cv2.resize(image, (new_h, new_w))
    
    return resized_image


def find_contours(image, scaling_factor, thresh_test=False):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(20,20))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    all_contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = all_contours[0] if len(all_contours) == 2 else all_contours[1]
    threshold_min_area = int(image.shape[0]*image.shape[1]/250)
    card_contours = []
    resized_card_contours = []

    for c in all_contours:
        area = cv2.contourArea(c)
        
        if area > threshold_min_area:
            c_scaled = c.astype(np.float32) / scaling_factor
            c_scaled = c_scaled.astype(np.int32)
            
            epsilon = 0.02 * cv2.arcLength(c_scaled, True)
            approx = cv2.approxPolyDP(c_scaled, epsilon, True)
            
            if len(approx) == 4:
                card_contours.append(approx)
                resized_card_contours.append(c)
                
    if thresh_test:
        return card_contours, resized_card_contours, thresh
    
    return card_contours, resized_card_contours
            

def draw_contours(image, contours, card_classes=None):
    
    for i, c in enumerate(contours):
        if card_classes != None:
            if card_classes[i] == "back":
                color = (12, 12, 255)
            elif card_classes.count(card_classes[i]) == 2:
                color = (12, 255, 12)
            else:
                color = (255, 12, 12)
        
        cv2.drawContours(image, [c], 0, color, image.shape[0]//100)
    

"""def crop_cards(image, contours):
    
    cards = []
        
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        
        cards.append(cv2.resize(image[y:y+h, x:x+w], (300, 300), interpolation=cv2.INTER_LINEAR))
    
    return cards"""
        

def crop_cards(image, contours, size=200):

    warped_cards = []

    destination_pts = np.array([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1]
    ], dtype="float32")

    for i, c in enumerate(contours):

        pts = c.reshape(4, 2).astype("float32")

        x_sorted = pts[np.argsort(pts[:, 0]), :]

        left_most = x_sorted[:2, :]
        right_most = x_sorted[2:, :]

        (top_l, bot_l) = left_most[np.argsort(left_most[:, 1]), :]

        (top_r, bot_r) = right_most[np.argsort(right_most[:, 1]), :]

        rect = np.array([top_l, top_r, bot_r, bot_l], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, destination_pts)

        warped = cv2.warpPerspective(image, M, (size, size))
        warped_cards.append(warped)

    return warped_cards
