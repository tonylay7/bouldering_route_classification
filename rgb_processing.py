from itertools import combinations
import colorsys
import cv2
import numpy as np
import scipy.cluster

def rgb_trueblack(rgb_values):
    """Check if the given RGB value is very close to the blackest black
    Args:
        rgb_values (tuple): (r, g, b)
    Returns:
        boolean: True if the given RGB value is very close to the blackest black
    """
    return (np.mean(rgb_values) <= 10)

def rgb_kmeans(img,num_clusters,bg_removed=True):
    """Perform K-Means clustering on an image with a specified number of clusters
    Args:
        img (Mat): input image
        num_clusters (int): number of clusters (K) for K-Means
    Returns:
        rgb_centres, index_max (tuple): rgb values for centres of clusters, index of the rgb centre with the highest frequency
    """
    # Re-format img
    ar = np.asarray(img)
    shape = ar.shape
    ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float)
    
    # Perform K-Means clustering
    rgb_centres, _ = scipy.cluster.vq.kmeans(ar, num_clusters)
    if bg_removed:
        rgb_centres = [c for c in rgb_centres if not rgb_trueblack(c)]
    vecs, _ = scipy.cluster.vq.vq(ar, rgb_centres)
    counts, bins = np.histogram(vecs, len(rgb_centres))
    
    # Index of the RGB array which holds the highest frequency
    index_max = np.argmax(counts)
    
    return (rgb_centres,index_max)

def rgb_similar(rgb_array,threshold=8):
    """Check if the R,G,B values in a given RGB array are similar to each other
    Args:
        rgb_array (array-like): (r,g,b)
        threshold: absolute difference threshold
    Returns:
        boolean: True if the R,G,B values in a given RGB array are similar to each other
    """
    abs_differences = [abs(a-b) for a, b in combinations(rgb_array, 2)]
    return (abs_differences[0] < threshold and abs_differences[1] < threshold and abs_differences[2] < threshold)

def rgb_thres(rgb_array,threshold,direction):
    """Check if the R,G,B values in a given RGB array are greater/lower than a specified threshold
    Args:
        rgb_array (array-like): (r,g,b)
        threshold (int): threshold value in range (0,256)
        direction (int): -1 for lower than threhsold, 1 for higher than threshold
    Returns:
        boolean: True if the R,G,B values in a given RGB array are greater/lower than a specified threshold
        
    """
    if direction == -1:
        return (rgb_array[0] < threshold and rgb_array[1] < threshold and rgb_array[2] < threshold)
    elif direction == 1:
        return (rgb_array[0] > threshold and rgb_array[1] > threshold and rgb_array[2] > threshold)
    else:
        raise Exception(f"The direction {direction} is not valid. It must be -1 or 1.")
    
def check_black_or_white(rgb_centres,idx_max):
    """Check if the R,G,B values in a given RGB array are black or white
    Args:
        rgb_centres (array-like): array of multiple (r,g,b) values
        idx_max (int): index of the RGB colour which had the highest frequency
    Returns:
        int: -1 if black, 0 if neither black or white, 1 if white 
    """
    
    # Specifically check for purple (since a lot of purple holds are smothered in chalk)
    top2 = np.argpartition(rgb_centres[idx_max], -2)[-2:]
    if (0 in top2 and 2 in top2) and (rgb_similar(rgb_centres[idx_max],22.5)):
        return 0
    else:
        # If the RGB values of the two centres are similar then this mostly comprises of white or black colours
        if len(rgb_centres) < 2:
            raise Exception(f'How has this happened? {rgb_centres}')
        if rgb_similar(rgb_centres[0],22.5) and rgb_similar(rgb_centres[1],22.5):
            # Check for black
            if (rgb_thres(rgb_centres[0],80,-1) and rgb_thres(rgb_centres[1],150,-1)):
                return -1
            elif (rgb_thres(rgb_centres[1],80,-1) and rgb_thres(rgb_centres[0],150,-1)):
                return -1
            # If it's not black or purple then it must be white
            else:
                return 1
        # Otherwise this is not black or white, it must be a different colour
        else:
            return 0

def color_from_hsv(h):
    """Return the colour based on the hue value
    Args:
        h: hue value
    Returns:
        (str): colour
    """
    if 0 <= h < 1.5:
        return 'red'
    # Some holds at this hue range are hard to distinguish into red or orange so return both
    elif 1.5 <= h < 3.9:
        return 'red','orange'
    elif 3.9 <= h < 20:
        return 'orange'
    elif 20 <= h < 40:
        return 'yellow'
    elif 40 <= h < 70:
        return 'green'
    elif 65 <= h < 125:
        return 'blue'
    elif 125 <= h < 166.6:
        return 'purple'
    elif 166.6 <= h < 174:
        return 'pink'
    # Some holds at this hue range are hard to distinguish into pink or red so return both
    elif 174 <= h < 178.6:
        return 'red','pink'
    else:
        return 'red'
        
def get_main_colour(subimg):
    """Return the main colour of an image using KMeans clusteirng
    Args:
        subimg: img source
    Returns:
        (str): colour
    """   
    # Convert image to RGB space
    img_rgb = cv2.cvtColor(subimg, cv2.COLOR_BGR2RGB)
    
    #show_image(img_rgb)
    
    # Perform K-Means clustering to split image into 3 main colours
    rgb_centres, idx_max = rgb_kmeans(img_rgb,num_clusters=3)
    
    # Retrieve the strongest RGB colour
    main_rgb = rgb_centres[idx_max]

    #print(main_rgb)
    
    # If the main RGB colour is black or white then return it
    if rgb_similar(main_rgb,10) and (rgb_close(main_rgb,60,31.5) or rgb_thres(main_rgb,40,-1)):
        #print("mainly black")
        return 'black'
    if rgb_similar(main_rgb,10) and rgb_thres(main_rgb,155,1):
        #print("mainly white")
        return 'white'
    
    # Otherwise perform a stronger check for whether it's black or white
    # Necessary as black holds can be covered in white chalk
    # or white holds can be turned blacker from wear over time
    bw_check = check_black_or_white(rgb_centres,idx_max)
    if bw_check == 1:
        #print("checked white")
        return 'white'
    elif bw_check == -1:
        #print("checked black")
        return 'black'
    
    # If the hold is not black or white then it must be a colour so find the colour
    else:
        # Normalise the rgb values
        (r, g, b) = (main_rgb[0] / 255, main_rgb[1] / 255, main_rgb[2] / 255)
        # Convert to HSV
        (h, _, _) = colorsys.rgb_to_hsv(r, g, b)
        # Return the colour from the HSV value
        return color_from_hsv(h*179)

def rgb_close(rgb_array, value, half_window=3):
    """Checks if the values R, G and B are close to a specified value
    Args:
        rgb_array (array-like): (r,g,b)
        value: goal value
        half_window: half size of window of closeness
    Returns:
        (str): colour
    """   
    return (value-half_window <= rgb_array[0] <= value+half_window and value-half_window <= rgb_array[1] <= value+half_window and value-half_window <= rgb_array[2] <= value+half_window)
