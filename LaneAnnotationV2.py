# Lane detection code for EGR 530

# Import helper functions file
from HelperFunctions import *
import warnings

# Suppress output from polyfit warning
warnings.simplefilter('ignore', np.RankWarning)

# Set the top and bottom points of the region of interest as a fraction of the frame (measured from the top)
topPointMultiplier = 0.25
bottomPointMultiplier = 0.9

# Set the number of bands between the bottom of the frame and the top point multiplier in which to find lane lines
measurementBands = 5

# Set the rate at which lane positions will update
laneUpdateRate = 0.1

# Variables to store latest coords of detected lane lines
steeringValue = 0
laneCoords = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])

# The video feed is read in as a VideoCapture object
cap = cv.VideoCapture("input2.mp4")
# cap = cv.VideoCapture(0)

while cap.isOpened():
    # Get current image
    ret, img = cap.read()

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

    # for b in range(measurementBands):
    b = 0

    # Apply crop
    img_edges_crop1, cropBoundaryCoords1 = RectangularMask(img_edges, 0.9, 0.99, 0.05, 0.45, 0.2, -0.03)
    img_edges_crop2, cropBoundaryCoords2 = RectangularMask(img_edges, 0.9, 0.99, 0.55, 0.95, -0.03, 0.2)
    img_edges_crop = img_edges_crop1 + img_edges_crop2

    # Show edges
    cv.imshow("Edges", ResizeFrame(img_edges_crop, 0.8))

    # Draw crop boundary on overlay
    DrawLines(overlay, cropBoundaryCoords1, (0, 0, 255))
    DrawLines(overlay, cropBoundaryCoords2, (0, 0, 255))
    DrawPointer(overlay, 0.5, (0, 0, 255))

    try:
        # Find lane lines
        leftLine, rightLine = FindLaneLines(img_edges_crop, 0.9, 1)

        # Update lane coords
        if np.count_nonzero(leftLine):
            laneCoords[0] = leftLine
        if np.count_nonzero(rightLine):
            laneCoords[1] = rightLine

        # Calculate steering value based on centers of lines
        leftLineCenter = (laneCoords[0][0] + laneCoords[0][2]) / (2 * img_edges_crop.shape[1])
        rightLineCenter = (laneCoords[1][0] + laneCoords[1][2]) / (2 * img_edges_crop.shape[1])
        steeringValue = (leftLineCenter + rightLineCenter) / 2

        # Draw steering value
        DrawText(overlay, "Steering: " + str(round(steeringValue, 3)), 0.98, (0, 255, 0))
        lines_found = True
    except ValueError:
        # If lane lines are not found
        DrawText(overlay, "Steering: error", 0.98, (0, 255, 0))
        lines_found = False

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
