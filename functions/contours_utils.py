
import math
import cv2
from matplotlib import pyplot as plt
import numpy as np

#https://stackoverflow.com/questions/55641425/check-if-two-contours-intersect @Ivan
def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Check if two contours intersects or if one is included into another
def contour_intersect(cnt_ref,cnt_query,DEBUG=False):

    ## Contour is a list of points

    # Check if contour is included
    if cv2.pointPolygonTest(cnt_ref,cnt_query[0][0].astype(np.uint8),False)>=0:
        if DEBUG:
            print("included")
        return True

    # Check if some points are equivalent
    for point_a in cnt_ref:
        for point_b in cnt_query:
            if (abs(point_a[0][0] - point_b[0][0]) <= 1 and point_a[0][1] == point_b[0][1]) or (point_a[0][0] == point_b[0][0] and abs(point_a[0][1] - point_b[0][1]) <= 1):
                if DEBUG:
                    print("border collapse")
                return True


    ## Connect each point to the following point to get a line
    ## If any of the lines intersect, then break
    for ref_idx in range(len(cnt_ref)-1):
    ## Create reference line_ref with point AB
        A = cnt_ref[ref_idx][0]
        B = cnt_ref[ref_idx+1][0] 
    
        for query_idx in range(len(cnt_query)-1):
            ## Create query line_query with point CD
            C = cnt_query[query_idx][0]
            D = cnt_query[query_idx+1][0]
        
            ## Check if line intersect
            if ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D):
                ## If true, break loop earlier
                if DEBUG:
                    print("border cut")
                return True

    return False

# Merge a given list of contour on a image size mask
def merge_contours(image,contours_to_merge,debug=False):
    canvas = np.zeros_like(image.astype(np.uint8))
    for contour in contours_to_merge:
        cv2.drawContours(canvas, [contour],-1, 255,-1)

    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if debug:
        canvas = np.zeros_like(image.astype(np.uint8))
        fig = plt.figure(figsize=(10,10))
        cv2.drawContours(canvas, contours,-1, 255,1)
        plt.imshow(canvas)
    if len(contours) > 0:
        #Case there is more than one contours (we merge contour that do not perfectly intersect)
        if len(contours)>1:
            if debug:
                print("more than 1 contour detected ("+str(len(contours))+")")
            areas = [cv2.contourArea(cnt) for cnt in contours]
            points = list()
            points = np.vstack(contours)
            (x, y), (MA, ma), angle = cv2.fitEllipse(points)
            # We try to fit an ellipse on theses contour. If the ellipse is too big (more than 3x the sum of detected contour), we return only the largest else we return the ellipse 
            if np.sum([cv2.contourArea(cnt) for cnt in contours]) * 3 < np.pi * MA * ma:
                canvas = np.zeros_like(image.astype(np.uint8))
                cv2.ellipse(canvas,[(x, y), (MA, ma), angle],255)
                contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                return contours[0]
            else:
                return contours[areas.index(max(areas))]
        return contours[0]
    print("Can't merge contours")
    return list()


