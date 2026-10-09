from enum import Enum
from threading import Lock


class DroneState(Enum):
    IDLE = "IDLE"
    FOLLOWING = "FOLLOWING"
    STOPPED = "STOPPED"
    EMERGENCY = "EMERGENCY"


class StateMachine:

    def __init__(self):
        self.state = DroneState.IDLE
        self.lock = Lock()

    def get_state(self):
        with self.lock:
            return self.state

    def process_command(self, command):
        with self.lock:

            if command == "emergency":
                self.state = DroneState.EMERGENCY

            elif self.state == DroneState.EMERGENCY:
                return "Emergency lock active. Manual reset required."

            elif command in ("follow", "resume"):
                self.state = DroneState.FOLLOWING

            elif command == "stop":
                self.state = DroneState.STOPPED

            elif command == "status":
                return f"Current state: {self.state.value}"

            else:
                return "Unknown command"

            return f"State changed to {self.state.value}"

    def reset_emergency(self):
        with self.lock:
            if self.state == DroneState.EMERGENCY:
                self.state = DroneState.IDLE
                return True
            return False