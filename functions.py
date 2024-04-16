import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity

#--------------------------------------------
def histogram_sim(_baseImg, _img):
    hist_baseImg = cv2.calcHist([_baseImg], [0], None, [256], [0, 256])
    hist_baseImg[255] = 0 
    cv2.normalize(hist_baseImg, hist_baseImg, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    hist_img = cv2.calcHist([_img], [0], None, [256], [0, 256])
    hist_img[255] = 0 
    cv2.normalize(hist_img, hist_img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    metric_val = cv2.compareHist(hist_baseImg, hist_img, cv2.HISTCMP_CORREL)
    return metric_val
#--------------------------------------------
def orb_sim(img1, img2):
    orb = cv2.ORB_create()
    kp_a, desc_a = orb.detectAndCompute(img1, None)
    kp_b, desc_b = orb.detectAndCompute(img2, None)

    if desc_a is None or desc_b is None or len(desc_a) == 0 or len(desc_b) == 0:
        return 0  
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc_a, desc_b)
    
    similar_regions = [i for i in matches if i.distance < 50]
    
    if len(matches) == 0:
        return 0
    
    return len(similar_regions) / len(matches)
#--------------------------------------------
def grid_image(_image, num):
    height, width = _image.shape
    cell_height = height // num
    cell_width = width // num

    cells = []

    for i in range(num):
        for j in range(num):
            x1 = j * cell_width
            y1 = i * cell_height
            x2 = (j + 1) * cell_width
            y2 = (i + 1) * cell_height

            cell = _image[y1:y2, x1:x2]
            cells.append(cell)

    return cells
#--------------------------------------------
def grid_similarity(_baseImage, _image, num):
    baseImg_cells = grid_image(_baseImage, num)
    img_cells = grid_image(_image, num)
    sum = 0

    fig, axes = plt.subplots(num, num, figsize=(8, 8))
    for i in range(num):
        for j in range(num):
            score, diff = structural_similarity(baseImg_cells[i * 5 + j], img_cells[i * 5 + j], full=True)
            histogram_similarity = histogram_sim(baseImg_cells[i * 5 + j], img_cells[i * 5 + j])
            sim = (score + histogram_similarity)/2
            sum = sim + sum
            #score = orb_sim(baseImg_cells[i * num + j], img_cells[i * num + j])
            axes[i, j].imshow(img_cells[i * num + j], cmap='gray')
            axes[i, j].set_title(f'SIMM: {score:.3f}\n Histogram: {histogram_similarity:.3f}',fontsize=8) 
            axes[i, j].axis('off')

    # plt.tight_layout()
    # plt.show()
    return sum

#--------------------------------------------