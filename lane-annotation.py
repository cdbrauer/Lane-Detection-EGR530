import cv2 as cv
import numpy as np
# import matplotlib.pyplot as plt

# Set the top point of the region of interest as a fraction of the frame (measured from the top)
top_point_multiplier = 0.5

# Find edges in a frame using canny edge detection
def detectEdges(frame):
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    # Applies a 5x5 gaussian blur with deviation of 0 to frame - not mandatory since Canny will do this for us
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    # Applies Canny edge detector with minVal of 50 and maxVal of 150
    frame_edges = cv.Canny(blur, 50, 150)
    # Return a frame showing all edges
    return frame_edges

# Crop a frame to the region of interest
def polygonMask(frame):
    # Gets the dimensions of the frame
    height = frame.shape[0]
    width = frame.shape[1]
    # Creates a triangular polygon for the mask defined by three (x, y) coordinates
    polygons = np.array([[(0, height), (width, height), (round(width*0.5), round(height*top_point_multiplier))]])
    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(frame)
    # Allows the mask to be filled with values of 1 and the other areas to be filled with values of 0
    cv.fillPoly(mask, polygons, 255)
    # A bitwise and operation between the mask and frame keeps only the triangular area of the frame
    frame_cropped = cv.bitwise_and(frame, mask)
    # Get the coords of the crop region edges
    edge_coords = np.array([[0, height, round(width*0.5), round(height*top_point_multiplier)], [width, height, round(width*0.5), round(height*top_point_multiplier)]])
    # Return frame with the mask applied
    return frame_cropped, edge_coords

# Calculate the end coordinates of a line given its slope and y intercept
def calculateEndCoordinates(frame, parameters):
    slope, intercept = parameters
    # Gets the dimensions of the frame
    height = frame.shape[0]
    width = frame.shape[1]
    # Sets initial y-coordinate as the bottom of the frame
    y1 = height
    # Sets final y-coordinate as the top of the region of interest
    y2 = round(height*top_point_multiplier)
    # Sets initial x-coordinate as (y1 - b) / m since y1 = mx1 + b
    x1 = int((y1 - intercept) / slope)
    # Sets final x-coordinate as (y2 - b) / m since y2 = mx2 + b
    x2 = int((y2 - intercept) / slope)
    # Create an array with the calculated coords and return it
    coords = np.array([x1, y1, x2, y2])
    return coords

# Find the left and right lane lines by averaging the detected edges
def findLaneLines(frame_edges):
    # Get the endpoints of every detected edge
    hough = cv.HoughLinesP(frame_edges, 2, np.pi / 180, 100, np.array([]), minLineLength = 100, maxLineGap = 50)
    # Empty arrays to store the coordinates of the left and right lines
    left = []
    right = []
    # Loops through every detected line
    if hough is not None:
        for line in hough:
            # Reshapes line from 2D array to 1D array
            x1, y1, x2, y2 = line.reshape(4)
            # Fits a linear polynomial to the x and y coordinates and returns a vector of coefficients which describe the slope and y-intercept
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            y_intercept = parameters[1]
            # If slope is negative, the line is to the left of the lane, and otherwise, the line is to the right of the lane
            if slope < 0:
                left.append((slope, y_intercept))
            else:
                right.append((slope, y_intercept))
        # Averages out all the values for left and right into a single slope and y-intercept value for each line
        left_avg = np.average(left, axis = 0)
        right_avg = np.average(right, axis = 0)
        # Calculates the x1, y1, x2, y2 coordinates for the left and right lines
        left_line = calculateEndCoordinates(frame_edges, left_avg)
        right_line = calculateEndCoordinates(frame_edges, right_avg)
        coords = np.array([left_line, right_line])
    else:
        coords = np.array([[], []])
    # Return the endpoint coords of the left and right lines
    return coords

# Draw overlay lines on a frame
def drawLines(frame, line_coords, color = (0, 255, 0)):
    # Creates an image filled with zero intensities with the same dimensions as the frame
    overlay = np.zeros_like(frame)
    # Checks if any lines are detected
    if line_coords is not None:
        for x1, y1, x2, y2 in line_coords:
            # Draws lines between two coordinates with green color and 5 thickness
            cv.line(overlay, (x1, y1), (x2, y2), color, 5)
    frame_overlay = cv.addWeighted(frame, 0.9, overlay, 1, 1)
    return frame_overlay

# Resize a frame by a scaling factor
def resizeFrame(frame, scale_factor):
    width = int(frame.shape[1] * scale_factor)
    height = int(frame.shape[0] * scale_factor)
    dim = (width, height)
    frame_resized = cv.resize(frame, dim, interpolation = cv.INTER_AREA)
    return frame_resized

# The video feed is read in as a VideoCapture object
#cap = cv.VideoCapture("input.mp4")
cap = cv.VideoCapture(0)

while (cap.isOpened()):
	# Get current frame
    # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    ret, frame = cap.read()
    
    # Apply edge detection
    frame_edges = detectEdges(frame)
    # Apply crop
    frame_edges_crop, crop_edge_coords = polygonMask(frame_edges)    
    # Overlay crop edges on frame
    frame_overlay = drawLines(frame, crop_edge_coords, (0, 0, 255))
    
    try:
        # Find lane lines
        line_coords = findLaneLines(frame_edges_crop)        
        # Overlay detected lane lines on frame
        frame_overlay = drawLines(frame_overlay, line_coords)
    except:
        pass
                
    # Open a new window and display the output frame
    cv.imshow("output", resizeFrame(frame_overlay, 0.8))

    # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
    if cv.waitKey(10) & 0xFF == ord('q'):
        break
        
# Frees up resources and closes all windows
cap.release()
cv.destroyAllWindows()
