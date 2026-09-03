import cv2
import speech_recognition as sr
import time

# =========================
# Face Detector
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =========================
# Speech Recognition
# =========================
r = sr.Recognizer()

WAKE_WORD = "drone"

# Prevents microphone from triggering every frame
last_listen_time = 0
LISTEN_COOLDOWN = 5  # seconds


# =========================
# Drone Command Processing
# =========================
def process_command(text):
    text = text.lower()

    if "come here" in text:
        print("ACTION: Move drone towards speaker")

    elif "stop" in text:
        print("ACTION: Stop drone safely")

    elif "hover" in text:
        print("ACTION: Hover drone at current position")

    elif "left" in text:
        print("ACTION: Turn drone left")

    elif "right" in text:
        print("ACTION: Turn drone right")

    else:
        print("Unknown command")


# =========================
# Open Webcam
# =========================
print("Starting camera...")

# macOS backend
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("ERROR: Could not open webcam")
    exit()

print("Camera started successfully")


# =========================
# Main Loop
# =========================
while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Failed to read frame")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4
    )

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    # Display camera feed
    cv2.imshow("Face Tracker + Voice Control", frame)

    # =========================
    # Voice Recognition
    # =========================
    current_time = time.time()

    if len(faces) > 0 and (current_time - last_listen_time > LISTEN_COOLDOWN):

        print("\nFace detected")
        print("Listening for voice command...")

        last_listen_time = current_time

        try:
            with sr.Microphone() as source:

                # Reduce background noise
                r.adjust_for_ambient_noise(source, duration=0.5)

                # Listen with timeout
                audio = r.listen(
                    source,
                    timeout=3,
                    phrase_time_limit=4
                )

                print("Recognizing speech...")

                text = r.recognize_google(audio).lower()

                print("You said:", text)

                # Wake word check
                if WAKE_WORD in text:
                    print("Wake word detected")
                    process_command(text)

                else:
                    print("Wake word not detected")

        except sr.WaitTimeoutError:
            print("No speech detected")

        except sr.UnknownValueError:
            print("Could not understand audio")

        except sr.RequestError as e:
            print("Speech recognition service error:", e)

        except Exception as e:
            print("Unexpected error:", e)

    # =========================
    # Quit Key
    # =========================
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================
# Cleanup
# =========================
cap.release()
cv2.destroyAllWindows()
print("Program closed")