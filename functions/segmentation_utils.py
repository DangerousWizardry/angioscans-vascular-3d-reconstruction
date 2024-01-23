import copy
import cv2
import pandas as pd
import numpy as np
import skimage


def extract_centroid_first_slice(img): 
    '''
    Le but de cette fonction est de segmenter l'aorte abdominale sur la première coupe de l'image,
    et de retourner les coordonnées du centroïde de la région segmentée.

    Args: 
        img (numpy array) : matrice qui représente l'image 3D

    Returns:
        coordinates (list[double]) : Retourne une liste avec les coordonnées du centroïde de l'artère
    '''
    z = img.shape[0] # numéro de la coupe
    zslice = img[z-1,:,:]

    # érosion de l'image
    zslice_erosion_shift = skimage.morphology.erosion(zslice, shift_x=True, shift_y=True)

    # Créer une image binaire à partir du seuil d'Otsu
    value_threshold_otsu = skimage.filters.threshold_otsu(zslice_erosion_shift)
    binary = np.where(zslice_erosion_shift < value_threshold_otsu, 0, 1)

    # Segmentation utilisant les composantes connectées sur l'image binaire
    label_array = skimage.measure.label(binary, background=0)

    # Enlever les objets inférieurs à une certaine taille
    labels_no_small_objects= skimage.morphology.remove_small_objects(label_array, min_size=150)

    # Récupérer les propriétés des régions segmentées
    properties_to_test=["label","centroid","eccentricity"]
    df_tot = pd.DataFrame(columns=properties_to_test)

    props = skimage.measure.regionprops_table(labels_no_small_objects, zslice_erosion_shift, properties=properties_to_test)
    df = pd.DataFrame(props)

    # Choisir l'artère comme étant la région segmentée avec la plus faible excentricité
    df_artere = df.loc[df['eccentricity'] == min(df.eccentricity)]
    centroid0 = df_artere['centroid-0'].iloc[0]
    centroid1 = df_artere['centroid-1'].iloc[0]

    coordinates = [centroid1, centroid0]

    return(coordinates) 

def find_knee_depth(image,bone_low_range=70,bone_high_range=255):
    depth = len(image)
    canvas = copy.deepcopy(image.astype(np.uint8))
    knee_low_index = int(depth/6)
    knee_high_index = int(depth/6 * 3)
    canvas[canvas>bone_high_range] = 0
    canvas[canvas<bone_low_range] = 0
    
    object_areas = list()

    for i in range(knee_low_index,knee_high_index):
        contours, _ = cv2.findContours(canvas[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_rects = [cv2.boundingRect(cnt) for cnt in contours]
        object_areas.append(max([w*h if (w>h and w/h < 3) or (w<h and h/w < 3) else 0 for (_,_,w,h) in bounding_rects],default=0))
    
    max_area_depth = knee_low_index + object_areas.index(max(object_areas))
    contours, _ = cv2.findContours(canvas[max_area_depth], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_rects = [cv2.boundingRect(cnt) for cnt in contours]
    filtered_bounding_rects = [w*h if (w/h < 3) else 0 for (_,_,w,h) in bounding_rects]
    index_contour = np.argsort(filtered_bounding_rects)[-2:]
    # Fill knee polygon
    mask = np.zeros_like(canvas[0], dtype=np.uint8)
    cv2.drawContours(mask, [contours[index_contour[0]]],-1, 255,1)
    cv2.drawContours(mask, [contours[index_contour[1]]],-1, 255,1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask = cv2.dilate(mask,kernel,iterations = 2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    knee_contour = [contours[0],contours[1]]
    return max_area_depth,knee_contour

def extract_arteries_position_from_knee(knee_image,knee_contour,low_filter=30):
    extraction_img = copy.deepcopy(knee_image.astype(np.uint8))
    knee_1,knee_2 = knee_contour
    extraction_img[extraction_img<low_filter]=0
    contours, _ = cv2.findContours(extraction_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidate = list()
    positions = list()
    i=0
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        rect_points = cv2.boxPoints(rect).astype(np.uint8)
        #We select only contour between 10 & 100 pixel "draft" areas that are not located in the knee contour
        if (10<np.prod(rect[1])<100 and cv2.contourArea(cnt)>0
        and cv2.pointPolygonTest(knee_1,rect[0],False)<0 
        and cv2.pointPolygonTest(knee_2,rect[0],False) < 0):
            candidate.append(cnt)
            positions.append([cv2.contourArea(cnt),i])
            i+=1
    positions.sort(key=lambda p: p[0],reverse=True)
    vessel1_contour = candidate[int(positions[0][1])]
    vessel2_contour = candidate[int(positions[1][1])]
    M = cv2.moments(vessel1_contour)
    vessel1 = (int(M['m10']/M['m00']),int(M['m01']/M['m00']))
    M = cv2.moments(vessel2_contour)
    vessel2 = (int(M['m10']/M['m00']),int(M['m01']/M['m00']))
    return vessel1,vessel2