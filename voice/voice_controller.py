import threading
import speech_recognition as sr


class VoiceController:

    def __init__(self, state_machine):
        self.state_machine = state_machine
        self.running = False
        self.thread = None

        self.recognizer = sr.Recognizer()

    def start(self):
        self.running = True

        self.thread = threading.Thread(
            target=self.listen_loop,
            daemon=True
        )

        self.thread.start()

    def stop(self):
        self.running = False

    def listen_loop(self):

        try:
            with sr.Microphone() as source:

                print("Calibrating microphone...")
                self.recognizer.adjust_for_ambient_noise(
                    source,
                    duration=1
                )

                print("Voice controller ready")

                while self.running:

                    try:
                        audio = self.recognizer.listen(
                            source,
                            timeout=1,
                            phrase_time_limit=4
                        )

                        command = self.recognizer.recognize_google(
                            audio
                        ).lower()

                        print(f"Heard: {command}")

                        self.handle_command(command)

                    except sr.WaitTimeoutError:
                        continue

                    except sr.UnknownValueError:
                        continue

                    except sr.RequestError as error:
                        print(f"Speech service error: {error}")

        except Exception as error:
            print(f"Microphone error: {error}")

    def handle_command(self, text):

        # Require the wake word.
        if "drone" not in text:
            return

        command = text.split("drone", 1)[1].strip()

        if "emergency stop" in command:
            action = "emergency"

        elif "stop" in command:
            action = "stop"

        elif "follow me" in command:
            action = "follow"

        elif "resume" in command:
            action = "resume"

        elif "status" in command:
            action = "status"

        else:
            print("Unrecognised drone command")
            return

        response = self.state_machine.process_command(action)

        print(f"Drone: {response}")