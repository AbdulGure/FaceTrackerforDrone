import csv
from datetime import datetime
from pathlib import Path


class PerformanceLogger:

    def __init__(self):

        project_root = Path(__file__).resolve().parent.parent
        log_directory = project_root / "logs"
        log_directory.mkdir(exist_ok=True)

        filename = datetime.now().strftime(
            "tracking_%Y%m%d_%H%M%S_%f.csv"
        )

        self.filepath = log_directory / filename

        self.file = open(
            self.filepath,
            "w",
            newline="",
            encoding="utf-8"
        )

        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "timestamp",
            "state",
            "identity_state",
            "tracking_id",
            "fps",
            "event"
        ])

        self.file.flush()

        print(f"Performance log: {self.filepath}")

    def log(
        self,
        state,
        identity_state,
        tracking_id,
        fps,
        event=""
    ):

        self.writer.writerow([
            datetime.now().isoformat(timespec="milliseconds"),
            state,
            identity_state,
            tracking_id if tracking_id is not None else "",
            round(fps, 2),
            event
        ])

    def close(self):

        if not self.file.closed:
            self.file.flush()
            self.file.close()