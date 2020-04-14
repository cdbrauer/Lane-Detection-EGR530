# Lane detection code for EGR 530

# Import helper functions file
from HelperFunctions import *
import warnings

# Suppress output from polyfit warning
warnings.simplefilter('ignore', np.RankWarning)

# Set the top and bottom points of the region of interest as a fraction of the frame (measured from the top)
top_point_multiplier = 0.25
bottom_point_multiplier = 0.9

# Set the rate at which lane positions will update
lane_update_rate = 0.1

# Variables to store latest coords of detected lane lines
steeringValueG = 0
laneCoordsG = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])
steeringValueGC = 0
laneCoordsGC = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])
steeringValueCombined = 0
laneCoordsCombined = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])

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
    img_edges = DetectEdges(img)
    img_recolor_edges = DetectEdges(img_recolor)

    # Apply crop
    (img_edges_crop, crop_boundary_coords) = TriangularMask(img_edges, top_point_multiplier, bottom_point_multiplier)
    (img_recolor_edges_crop, _) = TriangularMask(img_recolor_edges, top_point_multiplier, bottom_point_multiplier)

    # Draw crop boundary on overlay
    DrawLines(overlay, crop_boundary_coords, (0, 0, 255))
    DrawPointer(overlay, 0.5, (0, 0, 255))

    # Show recolored image
    # cv.imshow("Edges", ResizeFrame(img_edges_crop, 0.8))

    # Geometry only
    try:
        # Find lane lines
        leftLine, rightLine = FindLaneLinesHough(img_edges_crop, top_point_multiplier, 1)

        # Update lane coords
        if np.count_nonzero(leftLine):
            laneCoordsG[0] = leftLine
        if np.count_nonzero(rightLine):
            laneCoordsG[1] = rightLine

        # Calculate steering value based on centers of lines
        leftLineCenter = (laneCoordsG[0][0] + laneCoordsG[0][2]) / (2 * img_edges_crop.shape[1])
        rightLineCenter = (laneCoordsG[1][0] + laneCoordsG[1][2]) / (2 * img_edges_crop.shape[1])
        steeringValueG = (leftLineCenter + rightLineCenter) / 2

        # Draw steering value
        DrawText(overlay, " G: " + str(round(steeringValueG, 3)), 0.85, (0, 255, 0))
        gFound = True
    except ValueError:
        # If lane lines are not found
        DrawText(overlay, " G: error", 0.85, (0, 255, 0))
        gFound = False

    # Geometry + Color
    try:
        # Find lane lines
        leftLine, rightLine = FindLaneLinesHough(img_recolor_edges_crop, top_point_multiplier, 1)

        # Update lane coords
        if np.count_nonzero(leftLine):
            laneCoordsGC[0] = leftLine
        if np.count_nonzero(rightLine):
            laneCoordsGC[1] = rightLine

        # Calculate steering value based on centers of lines
        leftLineCenter = (laneCoordsGC[0][0] + laneCoordsGC[0][2]) / (2 * img_edges_crop.shape[1])
        rightLineCenter = (laneCoordsGC[1][0] + laneCoordsGC[1][2]) / (2 * img_edges_crop.shape[1])
        steeringValueGC = (leftLineCenter + rightLineCenter) / 2

        # Draw steering value
        DrawText(overlay, "GC: " + str(round(steeringValueGC, 3)), 0.9, (0, 255, 255))
        gcFound = True
    except ValueError:
        # If lane lines are not found
        DrawText(overlay, "GC: error", 0.9, (0, 255, 255))
        gcFound = False

    # Combine G and GC results
    if gFound and gcFound:
        laneCoordsCombined = lane_update_rate * (0.3 * laneCoordsG + 0.7 * laneCoordsGC) + (1 - lane_update_rate) * laneCoordsCombined
        steeringValueCombined = lane_update_rate * (0.3 * steeringValueG + 0.7 * steeringValueGC) + (1 - lane_update_rate) * steeringValueCombined
    elif gFound:
        laneCoordsCombined = lane_update_rate * (0.8 * laneCoordsG + 0.2 * laneCoordsGC) + (1 - lane_update_rate) * laneCoordsCombined
        steeringValueCombined = lane_update_rate * (0.8 * steeringValueG + 0.2 * steeringValueGC) + (1 - lane_update_rate) * steeringValueCombined
    elif gcFound:
        laneCoordsCombined = lane_update_rate * (0.0 * laneCoordsG + 1.0 * laneCoordsGC) + (1 - lane_update_rate) * laneCoordsCombined
        steeringValueCombined = lane_update_rate * (0.0 * steeringValueG + 1.0 * steeringValueGC) + (1 - lane_update_rate) * steeringValueCombined

    # Draw final steering value
    DrawText(overlay, " F: " + str(round(steeringValueCombined, 3)), 0.95, (255, 0, 0))

    # Draw detected lane lines
    DrawLines(overlay, laneCoordsG, (0, 255, 0))
    DrawPointer(overlay, steeringValueG, (0, 255, 0))
    DrawLines(overlay, laneCoordsGC, (0, 255, 255))
    DrawPointer(overlay, steeringValueGC, (0, 255, 255))
    DrawLines(overlay, laneCoordsCombined, (255, 0, 0))
    DrawPointer(overlay, steeringValueCombined, (255, 0, 0))

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