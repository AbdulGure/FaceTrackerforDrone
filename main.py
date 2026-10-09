import cv2
import time
import numpy as np

import multiprocessing as mp
import queue

from simulation.drone_simulator import simulator_process

from ultralytics import YOLO

from controller.follow_controller import FollowController

from controller.state_machine import (
    StateMachine,
    DroneState
)

from voice.voice_controller import VoiceController

from vision.face_recognition import (
    app,
    OWNER_FILE,
    SIMILARITY_THRESHOLD,
    cosine_similarity
)

from vision.target_association import (
    associate_face_to_person
)

from vision.identity_state import IdentityState

from monitoring.performance_logger import PerformanceLogger


# ==========================================
# CONFIGURATION
# ==========================================

VERIFY_INTERVAL = 0.5
VERIFIED_WINDOW = 0.75
MAX_UNVERIFIED_TIME = 2.0

YOLO_MODEL = "yolo11n.pt"

FPS_SMOOTHING = 0.1

# Record approximately 5 samples per second.
LOG_INTERVAL = 0.2


# ==========================================
# FPS
# ==========================================

def calculate_fps(previous_time, current_time, fps):

    delta = current_time - previous_time

    if delta <= 0:
        return fps

    instantaneous_fps = 1.0 / delta

    if fps == 0:
        return instantaneous_fps

    return (
        (1 - FPS_SMOOTHING) * fps
        + FPS_SMOOTHING * instantaneous_fps
    )


# ==========================================
# MAIN
# ==========================================

def main():

    if not OWNER_FILE.exists():
        print("Owner profile missing.")
        print("Run face enrollment first.")
        return

    owner_embedding = np.load(
        OWNER_FILE,
        allow_pickle=False
    )

    print("Loading YOLO...")

    model = YOLO(YOLO_MODEL)

    state_machine = StateMachine()
    voice = VoiceController(state_machine)
    follow_controller = FollowController()

    # Start the simulator in a separate process.
    sim_queue = mp.Queue(maxsize=2)

    sim_process = mp.Process(
        target=simulator_process,
        args=(sim_queue,),
        daemon=True
    )

    sim_process.start()

    cap = cv2.VideoCapture(
        0,
        cv2.CAP_AVFOUNDATION
    )

    if not cap.isOpened():
        raise RuntimeError("Camera unavailable")

    # --------------------------------------
    # TRACKING VARIABLES
    # --------------------------------------

    target_id = None

    last_verified = 0.0
    last_check = 0.0

    previous_state = DroneState.IDLE

    identity_state = IdentityState.LOST

    # --------------------------------------
    # PERFORMANCE VARIABLES
    # --------------------------------------

    fps = 0.0
    previous_frame_time = time.perf_counter()

    last_log_time = 0.0

    logger = PerformanceLogger()

    voice_started = False

    try:

        voice.start()
        voice_started = True

        print("\nF450 Voice + Vision Prototype")
        print("Say: Drone, follow me")
        print("Press F to follow")
        print("Press S to stop")
        print("Press E for emergency")
        print("Press R to reset emergency")
        print("Press Q to quit")

        follow_output = follow_controller.calculate(
            error_x=None,
            error_y=None,
            identity_verified=False
        )

        # ==================================
        # MAIN LOOP
        # ==================================

        while True:

            ret, frame = cap.read()

            if not ret:
                print("Camera unavailable.")
                break

            now = time.monotonic()

            height, width = frame.shape[:2]

            state = state_machine.get_state()

            follow_output = follow_controller.calculate(
                error_x=None,
                error_y=None,
                identity_verified=False
            )

            event = ""

            # ------------------------------
            # STATE TRANSITIONS
            # ------------------------------

            if state != previous_state:

                target_id = None

                last_verified = 0.0
                last_check = 0.0

                identity_state = IdentityState.LOST

                event = f"STATE_CHANGED_TO_{state.value}"

                previous_state = state

            people = []
            target_visible = False

            # ==================================
            # FOLLOWING MODE
            # ==================================

            if state == DroneState.FOLLOWING:

                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=[0],
                    conf=0.35,
                    verbose=False
                )

                for result in results:

                    if (
                        result.boxes is None
                        or result.boxes.id is None
                    ):
                        continue

                    boxes = result.boxes.xyxy.cpu().numpy()

                    ids = (
                        result.boxes.id
                        .int()
                        .cpu()
                        .tolist()
                    )

                    for box, track_id in zip(boxes, ids):

                        people.append({
                            "bbox": tuple(map(int, box)),
                            "id": track_id
                        })

                visible_ids = {
                    person["id"]
                    for person in people
                }

                # --------------------------
                # IMMEDIATE TARGET LOSS
                # --------------------------

                if (
                    target_id is not None
                    and target_id not in visible_ids
                ):

                    target_id = None
                    last_verified = 0.0

                    identity_state = IdentityState.LOST

                    event = "TARGET_DISAPPEARED"

                # --------------------------
                # FACE VERIFICATION
                # --------------------------

                if now - last_check >= VERIFY_INTERVAL:

                    last_check = now

                    faces = app.get(frame)

                    candidates = set()

                    for face in faces:

                        similarity = cosine_similarity(
                            owner_embedding,
                            face.normed_embedding
                        )

                        if similarity < SIMILARITY_THRESHOLD:
                            continue

                        person_box = associate_face_to_person(
                            face.bbox.astype(int),
                            people
                        )

                        if person_box is None:
                            continue

                        matches = [
                            person
                            for person in people
                            if person["bbox"] == person_box
                        ]

                        if len(matches) == 1:

                            candidates.add(
                                matches[0]["id"]
                            )

                    # ----------------------
                    # OWNER VERIFIED
                    # ----------------------

                    if len(candidates) == 1:

                        verified_id = next(
                            iter(candidates)
                        )

                        if target_id is None:

                            target_id = verified_id

                            last_verified = time.monotonic()

                            identity_state = IdentityState.VERIFIED

                            event = "OWNER_ACQUIRED"

                            print(
                                f"Owner acquired: ID {target_id}"
                            )

                        elif verified_id == target_id:

                            last_verified = time.monotonic()

                            identity_state = IdentityState.VERIFIED

                            event = "OWNER_VERIFIED"

                        else:

                            target_id = None
                            last_verified = 0.0

                            identity_state = IdentityState.LOST

                            event = "IDENTITY_CONFLICT"

                            print("Identity conflict detected.")

                    elif len(candidates) > 1:

                        target_id = None
                        last_verified = 0.0

                        identity_state = IdentityState.LOST

                        event = "AMBIGUOUS_IDENTITY"

                # --------------------------
                # IDENTITY STATE UPDATE
                # --------------------------

                identity_age = (
                    time.monotonic() - last_verified
                    if last_verified > 0
                    else float("inf")
                )

                if target_id is None:

                    identity_state = IdentityState.LOST

                elif identity_age > MAX_UNVERIFIED_TIME:

                    target_id = None
                    last_verified = 0.0

                    identity_state = IdentityState.LOST

                    event = "IDENTITY_EXPIRED"

                elif identity_age <= VERIFIED_WINDOW:

                    identity_state = IdentityState.VERIFIED

                else:

                    identity_state = IdentityState.UNVERIFIED

                # --------------------------
                # DRAW TRACKS
                # --------------------------

                for person in people:

                    x1, y1, x2, y2 = person["bbox"]

                    track_id = person["id"]

                    is_owner = (
                        target_id is not None
                        and track_id == target_id
                    )

                    if is_owner:
                        target_visible = True

                    if is_owner:

                        if identity_state == IdentityState.VERIFIED:
                            color = (0, 255, 0)
                        else:
                            color = (0, 255, 255)

                        label = (
                            f"OWNER ID {track_id}"
                        )

                    else:

                        color = (255, 0, 0)

                        label = (
                            f"PERSON ID {track_id}"
                        )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )

                    # ----------------------
                    # POSITION DIAGNOSTICS
                    # ----------------------

                    if is_owner:
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        # Horizontal tracking error
                        error_x = (
                                          center_x - width / 2
                                  ) / max(1, width / 2)

                        # Vertical tracking error
                        error_y = (
                                          center_y - height / 2
                                  ) / max(1, height / 2)

                        # ----------------------------------
                        # DISTANCE ESTIMATION
                        # ----------------------------------

                        # Height of the detected person
                        person_height = y2 - y1

                        # Proportion of the camera frame
                        # occupied by the detected person
                        height_ratio = person_height / max(1, height)

                        # ----------------------------------
                        # FOLLOW CONTROLLER
                        # ----------------------------------

                        follow_output = follow_controller.calculate(
                            error_x=error_x,
                            error_y=error_y,
                            identity_verified=(
                                    state == DroneState.FOLLOWING
                                    and identity_state == IdentityState.VERIFIED
                            ),
                            height_ratio=height_ratio
                        )

                        cv2.circle(
                            frame,
                            (center_x, center_y),
                            5,
                            (0, 0, 255),
                            -1
                        )

                        cv2.putText(
                            frame,
                            f"X: {error_x:.2f} Y: {error_y:.2f}",
                            (20, 85),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2
                        )

            # ==================================
            # SIMULATED DRONE OUTPUT
            # ==================================

            cv2.putText(
                frame,
                f"SIM YAW: {follow_output.yaw:+.2f}",
                (20, 165),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"SIM VERTICAL: {follow_output.vertical:+.2f}",
                (20, 205),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                follow_output.status,
                (20, 245),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            # ==================================
            # FPS CALCULATION
            # ==================================

            current_frame_time = time.perf_counter()

            fps = calculate_fps(
                previous_frame_time,
                current_frame_time,
                fps
            )

            previous_frame_time = current_frame_time

            # ==================================
            # SEND DATA TO SIMULATOR
            # ==================================

            sim_data = {
                "yaw": follow_output.yaw,
                "vertical": follow_output.vertical,
                "forward": follow_output.forward,
                "drone_state": state.value,
                "identity_state": identity_state.value,
            }

            # Keep the latest data instead of blocking
            # the camera processing loop.
            try:
                sim_queue.put_nowait(sim_data)

            except queue.Full:
                try:
                    sim_queue.get_nowait()
                except queue.Empty:
                    pass

                try:
                    sim_queue.put_nowait(sim_data)
                except queue.Full:
                    pass

            # ==================================
            # PERFORMANCE LOGGING
            # ==================================

            if (
                event
                or now - last_log_time >= LOG_INTERVAL
            ):

                logger.log(
                    state=state.value,
                    identity_state=identity_state.value,
                    tracking_id=target_id,
                    fps=fps,
                    event=event
                )

                last_log_time = now

            # ==================================
            # DISPLAY
            # ==================================

            cv2.putText(
                frame,
                f"STATE: {state.value}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"IDENTITY: {identity_state.value}",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (
                    (0, 255, 0)
                    if identity_state == IdentityState.VERIFIED
                    else (0, 255, 255)
                    if identity_state == IdentityState.UNVERIFIED
                    else (0, 0, 255)
                ),
                2
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "F450 Voice-Controlled Tracking",
                frame
            )

            # ==================================
            # KEYBOARD CONTROLS
            # ==================================

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("e"):
                state_machine.process_command("emergency")

            elif key == ord("r"):
                state_machine.reset_emergency()

            elif key == ord("f"):
                state_machine.process_command("follow")

            elif key == ord("s"):
                state_machine.process_command("stop")

    finally:

        if voice_started:
            voice.stop()

        # Stop simulator
        try:
            sim_queue.put_nowait(None)
        except queue.Full:
            pass

        sim_process.join(timeout=2)

        if sim_process.is_alive():
            sim_process.terminate()
            sim_process.join(timeout=2)

        sim_queue.close()

        cap.release()
        cv2.destroyAllWindows()

        logger.close()

        print(f"\nPerformance log saved: {logger.filepath}")


if __name__ == "__main__":
    main()