
import math
import cv2
import numpy as np


def regionSearch(image,point,searchRadius,minValue):
    search_memory = np.zeros_like(image, dtype=np.uint8)
    queue = [point]
    while queue:
        x, y = queue.pop(0)
        if search_memory[y, x] == 0:
            # Vérifier si la valeur du pixel est dans la plage acceptable
            # if point == (265,208):
            #     print(x, y)
            #     print(point)
            if int(image[y, x]) > minValue: #and int(image[y, x]) < maxValue:
                return (x,y)
            else:
                search_memory[y, x] = 1
            # Ajouter les voisins à la pile
            if abs(y-point[1]) < searchRadius+5 and abs(x-point[0]) < searchRadius+5:
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if 0 <= x + i < image.shape[1] and 0 <= y + j < image.shape[0]:
                            queue.append((x + i, y + j))
    #print("Radius was "+str(searchRadius)+" origin was "+str(point) +" and value was "+str(minValue))
    return (-1,-1)

def regionGrowing(image,point,nested_intensity,mean_radius=-1,alpha=0.60,exclusion_zone=None):

    segmented = np.zeros_like(image, dtype=np.uint8)

    # Initialiser la pile pour la croissance de la région
    stack = [point]

    # Récupérer la valeur du pixel du point de départ
    seed_value = nested_intensity.get_average()
    sum = 0
    if mean_radius > -1:
        unrealistic_area = 2 * np.pi * math.pow(mean_radius,2) * 2
    else:
        unrealistic_area = np.pow(10,7)

    while stack:
        x, y = stack.pop()
        # Vérifier si le pixel est déjà visité
        if segmented[y, x] == 0:
            # Vérifier si la valeur du pixel est dans la plage acceptable
            if image[y, x] >= seed_value*alpha:
                if isinstance(exclusion_zone,type(None)) or not any([cv2.pointPolygonTest(exclusion_zone[i] if len(exclusion_zone[i])>10 else np.array([[0,0],[1,1]]),(x,y),False) >= 0 for i in range(len(exclusion_zone))]):
                    # Ajouter le pixel à la région
                    segmented[y, x] = 255
                    sum += 1
                    # Ajouter les voisins à la pile
                    for i in range(-1, 2):
                        for j in range(-1, 2):
                            if 0 <= x + i < image.shape[1] and 0 <= y + j < image.shape[0]:
                                stack.append((x + i, y + j))

    
    contours, _ = cv2.findContours(segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours)>0:
        rect = cv2.minAreaRect(contours[0])
        contourBoundingWidth = max(rect[1])/2 #rect width
        # check if contour is big enough and thus anormal
        if contourBoundingWidth > 5 and (contourBoundingWidth > mean_radius * 3 or sum > unrealistic_area):
            if sum > 2000:
                #Bigger than the abdominal aorta
                return list()
            segmented = np.zeros_like(image, dtype=np.uint8)
            cv2.circle(segmented,point,int(mean_radius),255)
            contours, _ = cv2.findContours(segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            return contours[0]
        # else ok
        return contours[0]
    else:
        # print("target was "+str(point))
        # print("targeted intesity was"+str(seed_value))
        # print("not found")
        return list()