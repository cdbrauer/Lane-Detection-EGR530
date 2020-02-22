# Lane detection code for EGR 530

# Import helper functions file
from HelperFunctions import *

# Set the top point of the region of interest as a fraction of the frame (measured from the top)
top_point_multiplier = 0.25

# Set the rate at which lane positions will update
lane_update_rate = 0.4

# Variables to store coords of detected lane lines from current and previous iteration
steering_value = 0
lane_coords = np.array([[0,0,0,0], [0,0,0,0]])
steering_value_2 = 0
lane_coords_2 = np.array([[0,0,0,0], [0,0,0,0]])

# The video feed is read in as a VideoCapture object
# cap = cv.VideoCapture("input2.mp4")
cap = cv.VideoCapture(0)

while cap.isOpened():
    # Get current image
    ret, img = cap.read()

    # Initialize overlay
    overlay = initOverlay(img)

    # Convert to HSV and isolate yellow and white
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img_yellow = cv.inRange(img_hsv, (20, 39, 64), (35, 255, 255))
    img_white = cv.inRange(img_hsv, (0, 0, 229), (180, 38, 255))
    img_recolor = img_yellow + img_white

    # Show recolored image
    # cv.imshow("Recolored Frame", resizeFrame(img_recolor, 0.8))

    # Apply edge detection
    img_edges = detectEdges(img)
    img_recolor_edges = detectEdges(img_recolor)

    # Apply crop
    img_edges_crop, crop_boundary_coords = polygonMask(img_edges, top_point_multiplier)
    img_recolor_edges_crop, crop_boundary_coords_2 = polygonMask(img_recolor_edges, top_point_multiplier)
    midpoint = crop_boundary_coords[0][2]

    # Draw crop boundary on overlay
    drawLines(overlay, crop_boundary_coords, (0, 0, 255))
    drawPointer(overlay, midpoint, (0, 0, 255))

    # Geometry only
    try:
        # Find lane lines
        lane_coords_new, steering_value_new = findLaneLines(img_edges_crop, top_point_multiplier)
        # Update lane line coords using new measurement
        lane_coords = lane_update_rate*lane_coords_new + (1-lane_update_rate)*lane_coords
        steering_value = lane_update_rate*steering_value_new + (1-lane_update_rate)*steering_value
        # Draw steering value
        drawText(overlay, "G: " + str(int(steering_value)), 50, (0, 255, 0))
    except:
        # If lane lines are not found
        drawText(overlay, "G: error", 50, (0, 255, 0))

    # Geometry + color
    try:
        # Find lane lines
        lane_coords_2_new, steering_value_2_new = findLaneLines(img_recolor_edges_crop, top_point_multiplier)
        # Update lane line coords using new measurement
        lane_coords_2 = lane_update_rate*lane_coords_2_new + (1-lane_update_rate)* lane_coords_2
        steering_value_2 = lane_update_rate*steering_value_2_new + (1-lane_update_rate)*steering_value_2
        # Draw steering value
        drawText(overlay, "C: " + str(int(steering_value_2)), 10, (0, 255, 255))
    except:
        # If lane lines are not found
        drawText(overlay, "C: error", 10, (0, 255, 255))

    # Draw detected lane lines
    drawLines(overlay, lane_coords, (0, 255, 0))
    drawPointer(overlay, midpoint + steering_value, (0, 255, 0))
    drawLines(overlay, lane_coords_2, (0, 255, 255))
    drawPointer(overlay, midpoint+steering_value_2, (0, 255, 255))

    # Open a new window and display the output image with overlay
    frame_overlay = addOverlay(img, overlay)
    cv.imshow("Lane Detection", resizeFrame(frame_overlay, 0.8))

    # Read frames by intervals of 10 milliseconds
    # Break out of the while loop when the user presses the 'q' key
    if cv.waitKey(10) & 0xFF == ord('q'):
        break
        
# Free up resources and close all windows
cap.release()
cv.destroyAllWindows()
