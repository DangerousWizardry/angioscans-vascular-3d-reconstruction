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

    coordinates = [centroid0, centroid1]

    return(coordinates) 