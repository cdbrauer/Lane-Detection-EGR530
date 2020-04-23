# Lane detection code for EGR 530

# Import helper functions file
from HelperFunctions import *

# Set the number of measurement bands
measurementBands = 18

# Set the range of bands at which the steering value will be measured (inclusive)
testBandMin = 2 # 3
testBandMax = 7 # 8

# Set the bottom of the first measurement band as a fraction of the frame (measured from the top)
bottomPointMultiplier = 0.7

# Set the starting values for band height and floating band width
bandHeight = 0.04
bandWidth = 0.16

# Set the scale reduction between subsequent bands
scaleFalloff = 0.9

# Set values for tapering rectangles to compensate for perspective
taperOuter =  0.01
taperInner = -0.005

# Set the rate at which lane positions will update
lane_update_rate = 0.8

# Variables to store latest coords of detected lane lines
laneCoords = np.ones((measurementBands, 2, 4)) * 640

# The video feed is read in as a VideoCapture object
cap = cv.VideoCapture("videos/test1.mp4")
# cap = cv.VideoCapture(0)

while cap.isOpened():
    # Get current image
    ret, img = cap.read()

    # Get the dimensions of the frame
    height = img.shape[0]
    width = img.shape[1]

    # Initialize overlay
    overlay = InitOverlay(img)

    # Convert to HSV and isolate yellow and white
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img_yellow = cv.inRange(img_hsv, (20, 39, 64), (35, 255, 255))
    img_white = cv.inRange(img_hsv, (0, 0, 229), (180, 38, 255))
    img_recolor = img_yellow + img_white

    # Show recolored image
    # cv.imshow("Edges", ResizeFrame(img_recolor, 0.8))

    # Apply edge detection
    # img_edges = DetectEdges(img)
    img_edges = DetectEdges(img_recolor)

    # Initial mask location values
    currentBottom = bottomPointMultiplier
    currentTop = bottomPointMultiplier - bandHeight
    currentLL = 0.0
    currentLR = 0.48
    currentRL = 0.52
    currentRR = 1.0

    # Find lane lines in each measurement band
    for b in range(measurementBands):
        # Apply crop
        img_edges_cropL, cropBoundaryCoords1 = RectangularMask(img_edges, currentTop, currentBottom, currentLL, currentLR, taperOuter * (scaleFalloff**b), taperInner * (scaleFalloff**b))
        img_edges_cropR, cropBoundaryCoords2 = RectangularMask(img_edges, currentTop, currentBottom, currentRL, currentRR, taperInner * (scaleFalloff**b), taperOuter * (scaleFalloff**b))
        img_edges_crop = img_edges_cropL + img_edges_cropR

        # Draw crop boundary on overlay
        # DrawLines(overlay, cropBoundaryCoords1, (0, 0, 255))
        # DrawLines(overlay, cropBoundaryCoords2, (0, 0, 255))

        # Find left lane line
        try:
            laneCoordsL = FindLaneLineFit(img_edges_cropL, laneCoords[b][0], currentTop, currentBottom)
            laneCoords[b][0] = lane_update_rate*laneCoordsL + (1-lane_update_rate)*laneCoords[b][0]
        except IndexError:
            # print('No contours L')
            pass

        # Find right lane line
        try:
            laneCoordsR = FindLaneLineFit(img_edges_cropR, laneCoords[b][1], currentTop, currentBottom)
            laneCoords[b][1] = lane_update_rate*laneCoordsR + (1-lane_update_rate)*laneCoords[b][1]
        except IndexError:
            # print('No contours R')
            pass

        # Draw detected lane lines
        DrawLines(overlay, laneCoords[b], (0, 255, 0))

        currentBottom = currentTop
        currentTop = currentTop - bandHeight * (scaleFalloff ** b)
        currentLL = float(laneCoords[b][0][0]/width) - (bandWidth * (scaleFalloff ** b))
        currentLR = float(laneCoords[b][0][0]/width) + (bandWidth * (scaleFalloff ** b))
        currentRL = float(laneCoords[b][1][0]/width) - (bandWidth * (scaleFalloff ** b))
        currentRR = float(laneCoords[b][1][0]/width) + (bandWidth * (scaleFalloff ** b))

    # Calculate steering value based on centers of lines
    # laneCoords[bands][L/R][x1/y1/x2/y2]
    leftLineCenter = (np.average(laneCoords[testBandMin:testBandMax+1, 0, 0]) + np.average(laneCoords[testBandMin:testBandMax+1, 0, 2])) / (2 * width)
    rightLineCenter = (np.average(laneCoords[testBandMin:testBandMax+1, 1, 0]) + np.average(laneCoords[testBandMin:testBandMax+1, 1, 2])) / (2 * width)
    steeringValue = (leftLineCenter + rightLineCenter) / 2

    # Draw steering value
    DrawText(overlay, "Steering: " + str(round(steeringValue, 3)), 0.98, (0, 255, 0))
    DrawPointer(overlay, steeringValue, (0, 255, 0), 0.9)
    DrawPointer(overlay, 0.5, (0, 0, 255))

    # Open a new window and display the output image with overlay
    frame_overlay = AddOverlay(img, overlay)
    cv.imshow("Lane Detection", ResizeFrame(frame_overlay, 0.8))

    # Read frames by intervals of 10 milliseconds
    # Break out of the while loop when the user presses the 'q' key
    if cv.waitKey(10) & 0xFF == ord('q'):
        break
        
# Free up resources and close all windows
cap.release()
cv.destroyAllWindows()
