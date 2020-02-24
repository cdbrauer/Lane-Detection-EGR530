# Lane detection code for EGR 530

# Import helper functions file
from HelperFunctions import *

# Set the top and bottom points of the region of interest as a fraction of the frame (measured from the top)
top_point_multiplier = 0.25
bottom_point_multiplier = 0.9

# Set the rate at which lane positions will update
lane_update_rate = 0.1

# Variables to store latest coords of detected lane lines
steering_value_G = 0
lane_coords_G = np.array([[0,0,0,0], [0,0,0,0]])
steering_value_GC = 0
lane_coords_GC = np.array([[0,0,0,0], [0,0,0,0]])
steering_value_combined = 0
lane_coords_combined = np.array([[0,0,0,0], [0,0,0,0]])

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
    img_edges_crop, crop_boundary_coords_G, midpoint_G = polygonMask(img_edges, top_point_multiplier, bottom_point_multiplier)
    img_recolor_edges_crop, crop_boundary_coords_GC, midpoint_GC = polygonMask(img_recolor_edges, top_point_multiplier, bottom_point_multiplier)

    # Draw crop boundary on overlay
    drawLines(overlay, crop_boundary_coords_G, (0, 0, 255))
    drawPointer(overlay, midpoint_G, (0, 0, 255))

    # Geometry only
    try:
        # Find lane lines
        lane_coords_G, steering_value_G = findLaneLines(img_edges_crop, top_point_multiplier)
        # Draw steering value
        drawText(overlay, " G: " + str(int(steering_value_G)), 50, (0, 255, 0))
        g_found = True
    except:
        # If lane lines are not found
        drawText(overlay, " G: error", 50, (0, 255, 0))
        g_found = False

    # Geometry + Color
    try:
        # Find lane lines
        lane_coords_GC, steering_value_GC = findLaneLines(img_recolor_edges_crop, top_point_multiplier)
        # Draw steering value
        drawText(overlay, "GC: " + str(int(steering_value_GC)), 10, (0, 255, 255))
        gc_found = True
    except:
        # If lane lines are not found
        drawText(overlay, "GC: error", 10, (0, 255, 255))
        gc_found = False

    # Combine G and GC results
    if g_found and gc_found:
        lane_coords_combined = lane_update_rate * (0.3 * lane_coords_G + 0.7 * lane_coords_GC) + (1 - lane_update_rate) * lane_coords_combined
        steering_value_combined = lane_update_rate * (0.3 * steering_value_G + 0.7 * steering_value_GC) + (1 - lane_update_rate) * steering_value_combined
    elif g_found:
        lane_coords_combined = lane_update_rate * (0.8 * lane_coords_G + 0.2 * lane_coords_GC) + (1 - lane_update_rate) * lane_coords_combined
        steering_value_combined = lane_update_rate * (0.8 * steering_value_G + 0.2 * steering_value_GC) + (1 - lane_update_rate) * steering_value_combined
    elif gc_found:
        lane_coords_combined = lane_update_rate * (0.0 * lane_coords_G + 1.0 * lane_coords_GC) + (1 - lane_update_rate) * lane_coords_combined
        steering_value_combined = lane_update_rate * (0.0 * steering_value_G + 1.0 * steering_value_GC) + (1 - lane_update_rate) * steering_value_combined

    # Draw final steering value
    drawText(overlay, " F: " + str(int(steering_value_combined)), 90, (255, 0, 0))

    # Draw detected lane lines
    drawLines(overlay, lane_coords_G, (0, 255, 0))
    drawPointer(overlay, midpoint_G + steering_value_G, (0, 255, 0))
    drawLines(overlay, lane_coords_GC, (0, 255, 255))
    drawPointer(overlay, midpoint_GC + steering_value_GC, (0, 255, 255))
    drawLines(overlay, lane_coords_combined, (255, 0, 0))
    drawPointer(overlay, midpoint_G + steering_value_combined, (255, 0, 0))

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
