# importing the required modules  
import pytesseract  
import matplotlib.pyplot as plt  
import cv2  
import glob  
import os  
import numpy as np

# specifying the path to the number plate images folder as shown below  
file_path = "/home/rajat/Downloads/test_img/cropped_detected/4.jpg"  

# Find characters in the resulting images
img = cv2.imread(file_path)
def segment_characters(image) :

    # Preprocess cropped license plate image
    #img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # Make borders white
    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[72:75,:] = 255
    img_binary_lp[:,330:333] = 255

    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH/6,
                       LP_WIDTH/2,
                       LP_HEIGHT/10,
                       2*LP_HEIGHT/3]
    #plt.imshow(img_binary_lp, cmap='gray')
    #plt.show()
    cv2.imwrite('contour.jpg',img_binary_lp)
    return img_binary_lp



NP_list = []  
predicted_NP = []  
file_p = segment_characters(img)
for file_path in glob.glob('./contour.jpg', recursive = True):  
      
    NP_file = file_path.split("/")[-1]  
    number_plate, _ = os.path.splitext(NP_file)  
    '''  
    Here we will append the actual number plate to a list  
    '''  
    NP_list.append(number_plate)  
      
    '''  
    Reading each number plate image file using openCV  
    '''  
    NP_img = cv2.imread(file_path)  
      
    '''  
    We will then pass each number plate image file  
    to the Tesseract OCR engine utilizing the Python library  
    wrapper for it. We get back predicted_res for  
    number plate. We append the predicted_res in a  
    list and compare it with the original number plate  
    '''  
    predicted_res = pytesseract.image_to_string(NP_img, lang ='eng',config = '--oem 3 --psm 6 ')  #,  config ='--oem 3 --psm 6 -c tessedit_char_whitelist = ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
      
    filter_predicted_res = "".join(predicted_res.split()).replace(":", "").replace("-", "")  
    predicted_NP.append(filter_predicted_res)  
    

 







print("Original Number Plate", "\t", "Predicted Number Plate", "\t", "Accuracy")  
print("--------------------", "\t", "-----------------------", "\t", "--------")  
 
def estimate_predicted_accuracy(ori_list, pre_list):  
    for ori_plate, pre_plate in zip(ori_list, pre_list):  
        acc = "0 %"  
        number_matches = 0  
        if ori_plate == pre_plate:  
            acc = "100 %"  
        else:  
            if len(ori_plate) == len(pre_plate):  
                for o, p in zip(ori_plate, pre_plate):  
                    if o == p:  
                        number_matches += 1  
                acc = str(round((number_matches / len(ori_plate)), 2) * 100)  
                acc += "%"  
        print(ori_plate, "\t", pre_plate, "\t", acc)
        height, width, channels = img.shape 
        x,y,w,h = 0,20,height,width
        #image = cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 1)
        cv2.putText(img, pre_plate, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)
        cv2.imwrite('tested.jpg',img)
        cv2.imshow('License Plate text recognised', img)
        cv2.waitKey() #image will not show until this is called
        cv2.destroyWindow('License Plate text recognised')
        
          
  
estimate_predicted_accuracy(NP_list, predicted_NP)  
