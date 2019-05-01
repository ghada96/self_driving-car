import cv2
import numpy as num
import matplotlib.pyplot as pyp

img =cv2.imread('road.jpg')
grey_img=num.copy(img)
def canny(image):
    #convert image to greyscale
    grey=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    #reducing noise
    blur=cv2.GaussianBlur(grey,(5,5),0)
    canny=cv2.Canny(blur,50,150)
    return canny



#cv2.imshow('myPicture', canny)
def make_coordinates(image,line_parameters):
    slope,intercept=line_parameters
    y1=image.shape[0]
    y2=int(y1*(3/5))
    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)
    return num.array([x1,y1,x2,y2])
def average_slope_intercept(image,lines):
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1,y1,x2,y2=line.reshape(4)
        parameters=num.polyfit((x1,x2),(y1,y2),1)
        slope=parameters[0]
        intercept=parameters[1]
        if slope<0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_average=num.average(left_fit,axis=0)
    right_fit_average=num.average(right_fit,axis=0)
    left_line=make_coordinates(image,left_fit_average)
    right_line=make_coordinates(image,right_fit_average)
    return num.array([left_line,right_line])

def display_lines(image,lines):
    line_image=num.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line.reshape(4)

            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image

def region_of_interest(image):
    height=image.shape[0]
    shapes=num.array([
        [(200,height),(1100,height),(500,250)]
    ])
    mask=num.zeros_like(image)
    #255 : the color of the polygone is completely white
    cv2.fillPoly(mask,shapes,255)
    masked_image=cv2.bitwise_and(image,mask)
    return masked_image 

canny_image=canny(grey_img)
cropped_image=region_of_interest(canny_image)

lines=cv2.HoughLinesP(cropped_image,2,num.pi/180,100,num.array([]),minLineLength=40,maxLineGap=5)
#get a  single smooth line
averaged_lines=average_slope_intercept(grey_img,lines)
line_image=display_lines(grey_img,averaged_lines)
final_img=cv2.addWeighted(grey_img,0.8,line_image,1,1)
cv2.imshow("result",final_img)

#desplay the image infinetly until we press something on the keyboard
cv2.waitKey(0)
#cap = cv2.VideoCapture("video.mp4")
#while(cap.isOpened()):
    #ret , frame = cap.read()
  #  if ret:
       # canny_image=canny(frame)
        #cropped_image=region_of_interest(canny_image)
        #lines=cv2.HoughLinesP(cropped_image,2,num.pi/180,50,num.array([]),minLineLength=40,maxLineGap=4)
        #averaged_lines=average_slope_intercept(frame,lines)
       # line_image=display_lines(frame,averaged_lines)
       # final_img=cv2.addWeighted(frame,0.8,line_image,1,1)
       # cv2.imshow("result",final_img)
        #if cv2.waitKey(1)== ord('q'):
           # break
   # else:
       # break
#cap.release()
#cv2.destroyAllWindows()





