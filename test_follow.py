from controller.follow_controller import FollowController


controller = FollowController()


tests = [
    ("Centre", 0.0, 0.0, True),
    ("Right", 0.7, 0.0, True),
    ("Left", -0.7, 0.0, True),
    ("Above", 0.0, -0.7, True),
    ("Below", 0.0, 0.7, True),
    ("Identity lost", 0.7, 0.5, False),
]


for name, x, y, verified in tests:

    output = controller.calculate(
        error_x=x,
        error_y=y,
        identity_verified=verified
    )

    print(
        f"{name:15} "
        f"Yaw: {output.yaw:+.2f} "
        f"Vertical: {output.vertical:+.2f} "
        f"Status: {output.status}"
    )