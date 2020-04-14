import cv2 as cv
import numpy as np

# Find edges in a frame using canny edge detection
def DetectEdges(frame):
    try:
        # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    except cv.error:
        # Frame is already grayscale
        gray = frame
    # Applies a 5x5 gaussian blur with deviation of 0 to frame - not mandatory since Canny will do this for us
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    # Applies Canny edge detector with minVal of 50 and maxVal of 150
    frame_edges = cv.Canny(blur, 50, 150)
    # Return a frame showing all edges
    return frame_edges

# Crop a frame to the region of interest
def RectangularMask(frame, topPos, bottomPos, leftPos, rightPos, leftTaper = 0, rightTaper = 0):
    # Get the dimensions of the frame
    height = frame.shape[0]
    width = frame.shape[1]

    # Creates a polygon for the mask defined by four (x, y) coordinates
    polygons = np.array([[
        (round(width * (leftPos+leftTaper)), round(height * topPos)),
        (round(width * (rightPos-rightTaper)), round(height * topPos)),
        (round(width * rightPos), round(height * bottomPos)),
        (round(width * leftPos), round(height * bottomPos))
    ]])

    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(frame)

    # Allows the mask to be filled with values of 1 and the other areas to be filled with values of 0
    cv.fillPoly(mask, polygons, 255)

    # A bitwise and operation between the mask and frame keeps only the triangular area of the frame
    frame_cropped = cv.bitwise_and(frame, mask)

    # Get the coords of the crop region edges
    edgeCoords = np.array([
        [polygons[0][0][0], polygons[0][0][1], polygons[0][1][0], polygons[0][1][1]],
        [polygons[0][1][0], polygons[0][1][1], polygons[0][2][0], polygons[0][2][1]],
        [polygons[0][2][0], polygons[0][2][1], polygons[0][3][0], polygons[0][3][1]],
        [polygons[0][3][0], polygons[0][3][1], polygons[0][0][0], polygons[0][0][1]],
    ])

    # Return frame with the mask applied
    return frame_cropped, edgeCoords

# Crop a frame to the region of interest
def TriangularMask(frame, topPointPos, bottomPointPos):
    # Get the dimensions of the frame
    height = frame.shape[0]
    width = frame.shape[1]

    # Creates a triangular polygon for the mask defined by three (x, y) coordinates
    polygons = np.array([[
        (0, height),
        (width, height),
        (width, round(height * bottomPointPos)),
        (round(width*0.5), round(height * topPointPos)),
        (0, round(height * bottomPointPos))
    ]])

    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(frame)

    # Allows the mask to be filled with values of 1 and the other areas to be filled with values of 0
    cv.fillPoly(mask, polygons, 255)

    # A bitwise and operation between the mask and frame keeps only the triangular area of the frame
    frame_cropped = cv.bitwise_and(frame, mask)

    # Get the coords of the crop region edges
    edgeCoords = np.array([
        [0, height, width, height],
        [width, height, width, round(height * bottomPointPos)],
        [width, round(height * bottomPointPos), round(width * 0.5), round(height * topPointPos)],
        [round(width * 0.5), round(height * topPointPos), 0, round(height * bottomPointPos)],
        [0, round(height * bottomPointPos), 0, height],
    ])

    # Return frame with the mask applied
    return frame_cropped, edgeCoords

# Calculate the end coordinates of a line given its slope and y intercept
def CalculateEndCoordinates(frame, parameters, topPointPos, bottomPointPos):
    slope, intercept = parameters

    # Get the dimensions of the frame
    height = frame.shape[0]
    # width = frame.shape[1]

    # Sets initial and final y-coordinates
    y1 = round(height * bottomPointPos)
    y2 = round(height * topPointPos)

    # Sets initial x-coordinate as (y - b) / m since y = mx + b
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    # Create an array with the calculated coords and return it
    coords = np.array([x1, y1, x2, y2])
    return coords

# Find the left and right lane lines by averaging the detected edges
def FindLaneLines(frame_edges, topPointPos, bottomPointPos):
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
            yIntercept = parameters[1]

            # If slope is in the expected range (10deg to 80deg)
            if (abs(slope) > 0.36) and (abs(slope) < 5.67):
                # If slope is negative, the line is to the left of the lane, and otherwise, the line is to the right of the lane
                if slope < 0:
                    left.append((slope, yIntercept))
                else:
                    right.append((slope, yIntercept))

        if (len(left) == 0) and (len(right) == 0):
            raise ValueError("No lines in slope range")
        else:
            if len(left) > 0:
                # Average out all the values into a single slope and y-intercept value and calculate the x1, y1, x2, y2 coordinates
                leftAvg = np.average(left, axis = 0)
                leftLine = CalculateEndCoordinates(frame_edges, leftAvg, topPointPos, bottomPointPos)
            else:
                leftLine = np.array([0, 0, 0, 0])

            if len(right) > 0:
                # Average out all the values into a single slope and y-intercept value and calculate the x1, y1, x2, y2 coordinates
                rightAvg = np.average(right, axis = 0)
                rightLine = CalculateEndCoordinates(frame_edges, rightAvg, topPointPos, bottomPointPos)
            else:
                rightLine = np.array([0, 0, 0, 0])

            # Return the endpoint coords, steer value, and line center positions
            return leftLine, rightLine
    else:
        raise ValueError("No lines in frame")

# Draw lines on a frame
def DrawLines(frame, lineCoords, color = (255, 0, 0)):
    # Check if any lines are detected
    if lineCoords is not None:
        for x1, y1, x2, y2 in lineCoords:
            # Draw lines between two coordinates with color and 5 thickness
            cv.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 5)
    return frame

# Draw text on a frame
def DrawText(frame, text, pos = 0.95, color = (255, 0, 0)):
    # Get the dimensions of the frame
    height = frame.shape[0]
    # Add text
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(frame, str(text), (5,round(height*pos)), font, 1, color, 2, cv.LINE_AA)
    return frame

# Draw a small pointer at the bottom of a frame
def DrawPointer(frame, xPos, color = (255, 0, 0)):
    # Get the dimensions of the frame
    height = frame.shape[0]
    width = frame.shape[1]
    cv.line(frame, (int(width * xPos), height), (int(width * xPos), int(0.95 * height)), color, 5)
    return frame

def InitOverlay(frame):
    # Create an image filled with zero intensities with the same dimensions as the frame
    overlay = np.zeros_like(frame)
    return overlay

def AddOverlay(frame, overlay):
    frame_overlay = np.zeros_like(frame)
    cv.addWeighted(overlay, 0.9, frame, 1, 0, frame_overlay)
    return frame_overlay

# Resize a frame by a scaling factor
def ResizeFrame(frame, scale_factor):
    width = int(frame.shape[1] * scale_factor)
    height = int(frame.shape[0] * scale_factor)
    dim = (width, height)
    frame_resized = cv.resize(frame, dim, interpolation = cv.INTER_AREA)
    return frame_resized