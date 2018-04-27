import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import glob

from skimage.io import imread, imsave
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage import color



def detect_circles_in_image(image):
    edges = canny(image, sigma=3, low_threshold=10, high_threshold=50)

    hough_radii = np.arange(10, 60, 1)
    hough_res = hough_circle(edges, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, threshold=0.50,
                                               min_xdistance=50, min_ydistance=50)

    # Removes circles that are closer than 20px to any other circle
    acr = [accums[0]]; cxr = [cx[0]]; cyr = [cy[0]]; radiir = [radii[0]]
    for i in range(1, len(accums)): # For each point
        closest_than_20_to_any = False
        for j in range(0, len(radiir)): # For all already existing points
            if np.sqrt((cxr[j]-cx[i])**2 + (cyr[j]-cy[i])**2 ) < 20:
                closest_than_20_to_any = True
        if closest_than_20_to_any == False:
            acr.append(accums[i]); cxr.append(cx[i]) 
            cyr.append(cy[i]) ; radiir.append(radii[i])

    centers = np.transpose(np.array([cxr, cyr, radiir]))
    return centers


def measure_average_grayscale_in_circles(image, circles):
    x,y = np.meshgrid(range(0, image.shape[0]), range(0, image.shape[1]))
    vals = []
    if circles is not None:
        for c in circles:
            msk = (x-c[0])**2/(c[2]**2) + (y-c[1])**2/(c[2]**2) <= 1
            mv = float(np.sum(image*msk))/np.sum(msk)
            vals.append(mv)
    return vals

def generate_detection_control_image(image, circles):
    imagec = color.gray2rgb(image)
    for c in circles:
        center_y = c[1]
        center_x = c[0]
        radius = c[2]
        for re in np.arange(-3,3,1):
            circy, circx = circle_perimeter(int(center_y), int(center_x), int(radius+re))
            circy[circy<=0] = 0
            circx[circx<=0] = 0
            circy[circy>=1023] = 1023
            circx[circx>=1023] = 1023
            imagec[circy, circx] = (220, 20, 20)


    return imagec

def crop_image_at_circles(image, circles):
    rl = []
    if circles is not None:
        for c in circles:
            r = image[int(c[1])-64:int(c[1])+64, 
                      int(c[0])-64:int(c[0])+64]
            rl.append(r)
    return rl

def create_montage(rl):
    if rl is not None:
        n_c = len(rl)
        w = int(np.floor(np.sqrt(n_c)))
        if w**2 == n_c:
            h = w
        elif w*(w+1) >= n_c:
            h = w+1
        else:
            w = w+1
            h = w
        if len(rl[0].shape) ==  2:
            mtge = np.zeros((w*rl[0].shape[0], h*rl[0].shape[1]))
        else:
            mtge = np.zeros((w*rl[0].shape[0], h*rl[0].shape[1], rl[0].shape[2]))

        for n_im, im in enumerate(rl):
            i = int(np.floor(n_im/h))
            j = n_im - i*h
            mtge[i*im.shape[0]:(i+1)*im.shape[0], j*im.shape[1]:(j+1)*im.shape[1] ] = im
        return mtge
    return none


img_dir = 'data/acquisitions/'
circ_dir = 'data/circles/'
qc_dir = 'data/qc/'



# Part 1 of the pipeline - detect circles in the image and measure mean intensity    
# Loops through all the fields of view
for img_name in glob.glob(img_dir + "/*/*/*/*.png"):
    e, _ = os.path.split(img_name)    
    # Checks that the output directory structure exists and recreates it if not
    o_img_dir = e.replace(img_dir, circ_dir)
    o_qc_dir = e.replace(img_dir, qc_dir)
    for dd in [o_img_dir, o_qc_dir]:
        if not os.path.exists(dd):
            os.makedirs(dd)
    # Sets up the output paths
    circles_name = img_name.replace(img_dir, circ_dir).replace(".png", ".txt")
    qc_img_name = img_name.replace(img_dir,qc_dir)
    # If there is no output file, process the image
    if not os.path.exists(circles_name):
        print(img_name)
        img = imread(img_name, flags=0)
        circles = detect_circles_in_image(img)
        vals = measure_average_grayscale_in_circles(img, circles)
        if circles is not None and vals is not None:
            circles = np.c_[circles, vals]
        det_control_image = generate_detection_control_image(img, circles)
        imsave(qc_img_name, det_control_image)
        np.savetxt(circles_name, circles)

                        
# Part 2 of the pipeline - generate experiment control montages of the experiments
# Loops through the experiments
for e in glob.glob(circ_dir + "/*/*/*"):
    if not os.path.isdir(e):
        continue

    # Gets the subject directory
    subject_directory, e_dir = os.path.split(e)
    if not os.path.exists(subject_directory.replace(circ_dir, qc_dir)):
        os.makedirs(subject_directory.replace(circ_dir, qc_dir))            
    o_img = e.replace(circ_dir, qc_dir) + ".png"
    
    if not os.path.exists(o_img):
        all_crops = []
        for circ_name in glob.glob(e + "/*.txt"):
            circles = np.loadtxt(circ_name)
            if circles.shape[0] :
                img = plt.imread(circ_name
                                 .replace(circ_dir, img_dir)
                                 .replace(".txt",".png"))
                crops = crop_image_at_circles(img, circles)
                all_crops.extend(crops)
        mtg = create_montage(all_crops)

    #     print(o_img)
        imsave(o_img, mtg)
            

# Part 3 of the pipeline - aggregate the experiments
data = []
# Loop through all circles detected in the fields of view                
for circ_name in glob.glob(circ_dir + "/*/*/*/*.txt"):
    print(circ_name)
    circles = np.loadtxt(circ_name)
    if circles is not None:
        exp_path, fov = os.path.split(circ_name)
        sub_path, exp = os.path.split(exp_path)
        date_path, subject = os.path.split(sub_path)
        _, date = os.path.split(date_path)
        for i,c in enumerate(circles):
            l = [date, subject, exp, fov, i]
            l.extend(list(c))
            data.append(l)

dt = pd.DataFrame(data, columns=['Date','Subject','Experiment','FOV',
                                      'Circle','X','Y','R','Mean Intensity'])            

dt.to_csv('data/experiments.csv', index=False)




