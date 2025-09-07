import cv2
import numpy as np

def apply_cartoon_effect(video_capture):
    color_ranges = {
        "white": ((0, 0, 200), (180, 50, 255)),
        "red": ((0, 50, 50), (10, 255, 255)),
        "orange": ((11, 50, 50), (25, 255, 255)),
        "yellow": ((26, 50, 50), (35, 255, 255)),
        "green": ((36, 50, 50), (70, 255, 255)),
        "blue": ((71, 50, 50), (130, 255, 255)),
        "purple": ((131, 50, 50), (170, 255, 255)),
        "pink": ((171, 50, 50), (180, 255, 255)),
        "black": ((0,0,0), (180, 255, 30)),
    }
    #Defining the color ranges as a dictionary to standardise a range of colours for cartoonized effect
    #Ranges of HSV

    standard_colors = {
        "white": (255,255,255),
        "red": (0, 0, 150),
        "orange": (40, 159, 220),
        "yellow": (50, 165, 205),
        "green": (0, 100, 0),
        "blue": (100, 0, 0),
        "purple": (100, 0, 100),
        "pink": (255, 192, 203),
        "black": (0,0,0),
    }   
    #Set the standard colors for each range (BGR)

    kernel_size = (5, 5) #Set the kernel size for the blur effect

    ''' 
    Set the parameters for the bilateral filter
    Bilateral filter --- is highly effective at noise removal while preserving edges.
    It work like gaussian filter but more focus on edges
    '''
    bilateral_diameter = 9     
    bilateral_sigma_color = 75
    bilateral_sigma_space = 75

    while True:
        ret, frame = video_capture.read() #Read a frame from the webcam
        if not ret:
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0) #infinite loop
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Cannot read from video source")
                break
        
        original_frame = frame.copy()#copy of original frame    

        # Apply a median blur to reduce noise
        frame = cv2.medianBlur(frame, 5)

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)# Converting to HSV

        cartoon_frame = np.zeros_like(frame) #Create a blank black frame to hold the cartoonized effect

        # Apply the cartoonized effect
        for color, (lower, upper) in color_ranges.items(): #iterate through the colors
            mask = cv2.inRange(hsv_frame, np.array(lower), np.array(upper)) #creating a mask
            cartoon_frame[mask > 0] = standard_colors[color] #replacing colors in the blank black frame created

        cartoon_frame = cv2.bilateralFilter(cartoon_frame, bilateral_diameter, bilateral_sigma_color, bilateral_sigma_space)        
        # Apply a bilateral filter to reduce noise and smooth the image
        
        cartoon_frame = cv2.GaussianBlur(cartoon_frame, kernel_size, 0) 
        #Apply Guassian Blur. It is a uniform blur to reduce further noise in cartoon frame.        
        
        # Display the original and cartoonized frame
        cv2.imshow("Original Video", original_frame)
        cv2.imshow("Cartoonized Webcam", cartoon_frame)

        # Press 'esc' to quit
        if cv2.waitKey(1) == 27:
            break

    # Release the video capture object and close all windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Choose input source:")
    print("1. Webcam")
    print("2. Video file")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        video_capture = cv2.VideoCapture(0)
    elif choice == "2":
        video_path = input("Enter the path to your video file: ")
        video_capture = cv2.VideoCapture(video_path)
    else:
        print("Invalid choice. Exiting...")
        exit()

    if not video_capture.isOpened():
        print("Error: Could not open video source")
        exit()

    # Apply the cartoon effect
    apply_cartoon_effect(video_capture)