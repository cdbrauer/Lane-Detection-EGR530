# Lane detection code for EGR 530

# Import helper functions file
from HelperFunctions import *

# Set the top point of the region of interest as a fraction of the frame (measured from the top)
top_point_multiplier = 0.33

# The video feed is read in as a VideoCapture object
#cap = cv.VideoCapture("input.mp4")
cap = cv.VideoCapture(0)

while cap.isOpened():
    # Get current image
    ret, img = cap.read()

    # Apply edge detection
    img_edges = detectEdges(img)
    # Apply crop
    img_edges_crop, crop_boundary_coords = polygonMask(img_edges, top_point_multiplier)
    # Overlay crop boundary on image
    img_overlay = drawLines(img, crop_boundary_coords, (0, 0, 255))
    
    try:
        # Find lane lines
        lane_line_coords, steering_value = findLaneLines(img_edges_crop, top_point_multiplier)
        # Overlay detected lane lines on image
        img_overlay = drawLines(img_overlay, lane_line_coords)
    except:
        # If lane lines are not found
        steering_value = "error"
        
    # Print steering value
    drawText(img_overlay, steering_value)

    # Open a new window and display the output image
    cv.imshow("output", resizeFrame(img_overlay, 0.8))

    # Read frames by intervals of 10 milliseconds
    # Break out of the while loop when the user presses the 'q' key
    if cv.waitKey(10) & 0xFF == ord('q'):
        break
        
# Free up resources and close all windows
cap.release()
cv.destroyAllWindows()
