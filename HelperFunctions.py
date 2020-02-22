import cv2 as cv
import numpy as np

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
def polygonMask(frame, top_point_pos):
    # Gets the dimensions of the frame
    height = frame.shape[0]
    width = frame.shape[1]
    # Creates a triangular polygon for the mask defined by three (x, y) coordinates
    polygons = np.array([[(0, height), (width, height), (round(width*0.5), round(height * top_point_pos))]])
    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(frame)
    # Allows the mask to be filled with values of 1 and the other areas to be filled with values of 0
    cv.fillPoly(mask, polygons, 255)
    # A bitwise and operation between the mask and frame keeps only the triangular area of the frame
    frame_cropped = cv.bitwise_and(frame, mask)
    # Get the coords of the crop region edges
    edge_coords = np.array([[0, height, round(width*0.5), round(height * top_point_pos)], [width, height, round(width * 0.5), round(height * top_point_pos)]])
    # Return frame with the mask applied
    return frame_cropped, edge_coords

# Calculate the end coordinates of a line given its slope and y intercept
def calculateEndCoordinates(frame, parameters, top_point_pos):
    slope, intercept = parameters
    # Gets the dimensions of the frame
    height = frame.shape[0]
    # width = frame.shape[1]
    # Sets initial y-coordinate as the bottom of the frame
    y1 = height
    # Sets final y-coordinate as the top of the region of interest
    y2 = round(height * top_point_pos)
    # Sets initial x-coordinate as (y1 - b) / m since y1 = mx1 + b
    x1 = int((y1 - intercept) / slope)
    # Sets final x-coordinate as (y2 - b) / m since y2 = mx2 + b
    x2 = int((y2 - intercept) / slope)
    # Create an array with the calculated coords and return it
    coords = np.array([x1, y1, x2, y2])
    return coords

# Find the left and right lane lines by averaging the detected edges
def findLaneLines(frame_edges, top_point_pos):
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
            # If slope is in the expected range (30deg to 70deg)
            if (abs(slope) > 0.6) and (abs(slope) < 2.7):
                # If slope is negative, the line is to the left of the lane, and otherwise, the line is to the right of the lane
                if slope < 0:
                    left.append((slope, y_intercept))
                else:
                    right.append((slope, y_intercept))
        # Average out all the values for left and right into a single slope and y-intercept value for each line
        left_avg = np.average(left, axis = 0)
        right_avg = np.average(right, axis = 0)
        # Calculates the x1, y1, x2, y2 coordinates for the left and right lines
        left_line = calculateEndCoordinates(frame_edges, left_avg, top_point_pos)
        right_line = calculateEndCoordinates(frame_edges, right_avg, top_point_pos)
        coords = np.array([left_line, right_line])
        # Calculate steering value based on centers of lines
        left_line_center = (left_line[0] + left_line[2])/2
        right_line_center = (right_line[0] + right_line[2])/2
        steer = ((left_line_center + right_line_center)/2) - (frame_edges.shape[1]/2)
    else:
        coords = np.array([[], []])
        steer = 0
    # Return the endpoint coords of the left and right lines
    return coords, steer

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

# Draw overlay text on a frame
def drawText(frame, text):
    # Gets the dimensions of the frame
    height = frame.shape[0]
    # width = frame.shape[1]
    # Add text
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(frame, str(text), (5,height-5), font, 1, (0,255,0), 2, cv.LINE_AA)

# Resize a frame by a scaling factor
def resizeFrame(frame, scale_factor):
    width = int(frame.shape[1] * scale_factor)
    height = int(frame.shape[0] * scale_factor)
    dim = (width, height)
    frame_resized = cv.resize(frame, dim, interpolation = cv.INTER_AREA)
    return frame_resized