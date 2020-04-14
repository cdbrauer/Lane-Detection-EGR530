# Lane detection code for EGR 530

# Import helper functions file
from HelperFunctions import *
import warnings

# Suppress output from polyfit warning
warnings.simplefilter('ignore', np.RankWarning)

# Set the top and bottom points of the region of interest as a fraction of the frame (measured from the top)
topPointMultiplier = 0.25
bottomPointMultiplier = 0.98

# Set the number of bands between the bottom point multiplier and the top point multiplier in which to find lane lines
measurementBands = 5

# Set the rate at which lane positions will update
lane_update_rate = 0.2

# Variables to store latest coords of detected lane lines
steeringValue = 0
laneCoords = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])

# The video feed is read in as a VideoCapture object
cap = cv.VideoCapture("input2.mp4")
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

    # Apply edge detection
    # img_edges = DetectEdges(img)
    img_edges = DetectEdges(img_recolor)

    # Show edges
    # cv.imshow("Edges", ResizeFrame(img_edges_crop, 0.8))

    # Find lane lines in each measurement band
    # for b in range(measurementBands):
    # b = 0
    currentTop = 0.9
    currentBottom = 0.98

    # Apply crop
    img_edges_cropL, cropBoundaryCoords1 = RectangularMask(img_edges, currentTop, currentBottom, 0.05, 0.45, 0.2, -0.03)
    img_edges_cropR, cropBoundaryCoords2 = RectangularMask(img_edges, currentTop, currentBottom, 0.55, 0.95, -0.03, 0.2)
    img_edges_crop = img_edges_cropL + img_edges_cropR

    # Draw crop boundary on overlay
    DrawLines(overlay, cropBoundaryCoords1, (0, 0, 255))
    DrawLines(overlay, cropBoundaryCoords2, (0, 0, 255))
    DrawPointer(overlay, 0.5, (0, 0, 255))

    # Find left lane line
    try:
        laneCoordsL = FindLaneLineFit(img_edges_cropL, laneCoords[0], currentTop, currentBottom)
        laneCoords[0] = lane_update_rate*laneCoordsL + (1-lane_update_rate)*laneCoords[0]
    except IndexError:
        print('No contours L')

    # Find right lane line
    try:
        laneCoordsR = FindLaneLineFit(img_edges_cropR, laneCoords[1], currentTop, currentBottom)
        laneCoords[1] = lane_update_rate*laneCoordsR + (1-lane_update_rate)*laneCoords[1]
    except IndexError:
        print('No contours R')

    # Calculate steering value based on centers of lines
    leftLineCenter = (laneCoords[0][0] + laneCoords[0][2]) / (2 * img_edges_crop.shape[1])
    rightLineCenter = (laneCoords[1][0] + laneCoords[1][2]) / (2 * img_edges_crop.shape[1])
    steeringValue = (leftLineCenter + rightLineCenter) / 2

    # Draw steering value
    DrawText(overlay, "Steering: " + str(round(steeringValue, 3)), 0.98, (0, 255, 0))

    # Draw detected lane lines
    DrawLines(overlay, laneCoords, (0, 255, 0))
    DrawPointer(overlay, steeringValue, (0, 255, 0))

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
