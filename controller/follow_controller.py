from dataclasses import dataclass


@dataclass
class FollowOutput:
    yaw: float = 0.0
    vertical: float = 0.0
    forward: float = 0.0
    status: str = "HOLD"


class FollowController:
    """
    Diagnostic simulated following controller.

    Outputs are normalized simulation values,
    NOT real flight commands.
    """

    def __init__(
        self,
        kp_yaw=0.6,
        kp_vertical=0.4,
        kp_forward=0.8,
        deadzone=0.12,
        distance_deadzone=0.08,
        target_height_ratio=0.55,
        max_output=0.5,
    ):
        self.kp_yaw = kp_yaw
        self.kp_vertical = kp_vertical
        self.kp_forward = kp_forward

        self.deadzone = deadzone
        self.distance_deadzone = distance_deadzone
        self.target_height_ratio = target_height_ratio
        self.max_output = max_output

    def clamp(self, value):
        return max(
            -self.max_output,
            min(self.max_output, value)
        )

    def calculate(
        self,
        error_x,
        error_y,
        identity_verified,
        height_ratio=None,
    ):
        if not identity_verified:
            return FollowOutput(
                status="HOLD - IDENTITY NOT VERIFIED"
            )

        if error_x is None or error_y is None:
            return FollowOutput(
                status="HOLD - TARGET NOT VISIBLE"
            )

        yaw = 0.0
        vertical = 0.0
        forward = 0.0

        # Horizontal alignment
        if abs(error_x) >= self.deadzone:
            yaw = self.clamp(
                self.kp_yaw * error_x
            )

        # Vertical alignment
        if abs(error_y) >= self.deadzone:
            vertical = self.clamp(
                -self.kp_vertical * error_y
            )

        # Relative distance control
        if height_ratio is not None:

            distance_error = (
                self.target_height_ratio
                - height_ratio
            )

            if abs(distance_error) >= self.distance_deadzone:
                forward = self.clamp(
                    self.kp_forward * distance_error
                )

        if yaw == 0 and vertical == 0 and forward == 0:
            status = "TARGET CENTRED"
        else:
            status = "SIMULATED FOLLOWING"

        return FollowOutput(
            yaw=yaw,
            vertical=vertical,
            forward=forward,
            status=status,
        )