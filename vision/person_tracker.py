import cv2

# ==========================================
# CAMERA CONFIGURATION
# ==========================================

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    raise RuntimeError("Could not access camera")

# ==========================================
# TRACKER CONFIGURATION
# ==========================================

def create_tracker():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()

    if hasattr(cv2, "legacy"):
        return cv2.legacy.TrackerCSRT_create()

    raise RuntimeError(
        "CSRT unavailable. Install opencv-contrib-python."
    )


tracker = None
tracking = False

print("F450 PERSON TRACKING SYSTEM")
print("--------------------------")
print("Press S to select a person")
print("Press R to reset tracking")
print("Press Q to quit")

# ==========================================
# MAIN CAMERA LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read camera frame")
        break

    frame_height, frame_width = frame.shape[:2]

    frame_center_x = frame_width // 2
    frame_center_y = frame_height // 2

    # Draw camera centre
    cv2.drawMarker(
        frame,
        (frame_center_x, frame_center_y),
        (255, 0, 0),
        cv2.MARKER_CROSS,
        20,
        2
    )

    # ======================================
    # UPDATE TRACKER
    # ======================================

    if tracking:

        success, box = tracker.update(frame)

        if success:

            x, y, w, h = [int(v) for v in box]

            target_center_x = x + w // 2
            target_center_y = y + h // 2

            # Calculate position difference
            error_x = target_center_x - frame_center_x
            error_y = target_center_y - frame_center_y

            # Normalize position
            normalized_x = error_x / frame_center_x
            normalized_y = error_y / frame_center_y

            # Draw target bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Draw target centre
            cv2.circle(
                frame,
                (target_center_x, target_center_y),
                5,
                (0, 0, 255),
                -1
            )

            # Draw line from camera to target
            cv2.line(
                frame,
                (frame_center_x, frame_center_y),
                (target_center_x, target_center_y),
                (255, 255, 0),
                2
            )

            # ==================================
            # MOVEMENT DECISION
            # ==================================

            tolerance = 0.15

            if normalized_x < -tolerance:
                horizontal = "TARGET LEFT"

            elif normalized_x > tolerance:
                horizontal = "TARGET RIGHT"

            else:
                horizontal = "CENTRED"

            if normalized_y < -tolerance:
                vertical = "TARGET ABOVE"

            elif normalized_y > tolerance:
                vertical = "TARGET BELOW"

            else:
                vertical = "CENTRED"

            movement = f"{horizontal} | {vertical}"

            # Display movement
            cv2.putText(
                frame,
                movement,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"X Error: {normalized_x:.2f}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Y Error: {normalized_y:.2f}",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        else:

            print("Target lost")

            tracking = False
            tracker = None

    # ======================================
    # DISPLAY STATUS
    # ======================================

    status = "TRACKING" if tracking else "NO TARGET"

    cv2.putText(
        frame,
        f"STATUS: {status}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0) if tracking else (0, 0, 255),
        2
    )

    cv2.imshow("F450 Person Tracking", frame)

    # ======================================
    # KEYBOARD CONTROLS
    # ======================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):

        print("Select target")

        box = cv2.selectROI(
            "F450 Person Tracking",
            frame,
            fromCenter=False,
            showCrosshair=True
        )

        if box[2] > 0 and box[3] > 0:

            tracker = create_tracker()
            tracker.init(frame, box)

            tracking = True

            print("Target selected")
            print("Tracking started")

    elif key == ord("r"):

        tracker = None
        tracking = False

        print("Tracking reset")

    elif key == ord("q"):
        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()
cv2.destroyAllWindows()

print("Tracking system stopped")