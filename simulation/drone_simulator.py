import math
import queue
import time




WIDTH = 900
HEIGHT = 650

BACKGROUND = (20, 24, 35)
WHITE = (240, 240, 240)
GREEN = (50, 220, 120)
YELLOW = (255, 200, 50)
RED = (240, 75, 75)
BLUE = (75, 150, 240)
GREY = (110, 120, 135)


def draw_drone(screen, x, y, angle):
    import pygame
    """Draw a top-down quadcopter."""

    arm_length = 65
    motor_radius = 20

    # X-shaped quadcopter
    for offset in (45, 135, 225, 315):
        radians = math.radians(angle + offset)

        motor_x = x + math.cos(radians) * arm_length
        motor_y = y + math.sin(radians) * arm_length

        pygame.draw.line(
            screen,
            GREY,
            (x, y),
            (motor_x, motor_y),
            10
        )

        pygame.draw.circle(
            screen,
            BLUE,
            (int(motor_x), int(motor_y)),
            motor_radius,
            4
        )

    pygame.draw.circle(
        screen,
        WHITE,
        (int(x), int(y)),
        22
    )

    # Heading indicator
    radians = math.radians(angle - 90)

    heading_x = x + math.cos(radians) * 35
    heading_y = y + math.sin(radians) * 35

    pygame.draw.line(
        screen,
        GREEN,
        (x, y),
        (heading_x, heading_y),
        5
    )


def simulator_process(data_queue):
    """
    Runs Pygame in its own process.

    Receives diagnostic controller outputs
    from the webcam application.
    """
    import pygame

    pygame.init()

    screen = pygame.display.set_mode(
        (WIDTH, HEIGHT)
    )

    pygame.display.set_caption(
        "F450 Drone Simulator"
    )

    font = pygame.font.SysFont(
        "Arial",
        24
    )

    small_font = pygame.font.SysFont(
        "Arial",
        19
    )

    clock = pygame.time.Clock()

    angle = 0.0
    altitude = 0.0

    yaw = 0.0
    vertical = 0.0
    forward = 0.0

    # Virtual drone position
    drone_x = WIDTH / 2
    drone_y = HEIGHT / 2

    # Diagnostic movement speed
    MOVE_SPEED = 180.0

    drone_state = "IDLE"
    identity_state = "LOST"

    running = True
    last_update = time.monotonic()

    while running:

        dt = min(
            clock.tick(60) / 1000.0,
            0.05
        )

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

        # Get newest tracking information
        try:
            while True:
                data = data_queue.get_nowait()

                if data is None:
                    running = False
                    break

                yaw = float(data.get("yaw", 0.0))
                vertical = float(data.get("vertical", 0.0))
                forward = float(data.get("forward", 0.0))

                drone_state = data.get(
                    "drone_state",
                    "IDLE"
                )

                identity_state = data.get(
                    "identity_state",
                    "LOST"
                )

                last_update = time.monotonic()

        except queue.Empty:
            pass

        # Stop applying commands if updates stop.
        if time.monotonic() - last_update > 0.5:
            yaw = 0.0
            vertical = 0.0
            forward = 0.0

        # Diagnostic outputs only.
        if (
            drone_state != "FOLLOWING"
            or identity_state != "VERIFIED"
        ):
            yaw = 0.0
            vertical = 0.0

        # ----------------------------------
        # SIMULATED YAW VISUALISATION
        # ----------------------------------

        # Convert yaw input into a target display angle.
        # This is a visual indicator, not a physical yaw model.
        target_angle = yaw * 90.0

        # Smoothly approach the target angle.
        rotation_speed = 4.0

        angle += (
                         target_angle - angle
                 ) * min(1.0, rotation_speed * dt)
        # Heading: 0 degrees points upwards.
        heading = math.radians(angle - 90)

        # Move along the drone's current heading.
        drone_x += (
                math.cos(heading)
                * forward
                * MOVE_SPEED
                * dt
        )

        drone_y += (
                math.sin(heading)
                * forward
                * MOVE_SPEED
                * dt
        )

        # Keep the drone inside the simulation window.
        drone_x = max(
            90,
            min(WIDTH - 90, drone_x)
        )

        drone_y = max(
            90,
            min(HEIGHT - 90, drone_y)
        )

        # Simulated vertical response
        altitude += vertical * 2.0 * dt
        altitude = max(0.0, min(20.0, altitude))

        screen.fill(BACKGROUND)

        title = font.render(
            "F450 VIRTUAL DRONE",
            True,
            WHITE
        )

        screen.blit(
            title,
            (30, 25)
        )

        draw_drone(
            screen,
            drone_x,
            drone_y,
            angle
        )

        info = [
            f"Drone state: {drone_state}",
            f"Identity: {identity_state}",
            f"Yaw input: {yaw:+.2f}",
            f"Vertical input: {vertical:+.2f}",
            f"Heading: {angle % 360:.1f} degrees",
            f"Virtual altitude: {altitude:.2f} m",
            f"Forward input: {forward:+.2f}",
        ]

        for index, line in enumerate(info):

            text = small_font.render(
                line,
                True,
                WHITE
            )

            screen.blit(
                text,
                (30, 85 + index * 32)
            )

        if identity_state == "VERIFIED":
            indicator_color = GREEN

        elif identity_state == "UNVERIFIED":
            indicator_color = YELLOW

        else:
            indicator_color = RED

        pygame.draw.circle(
            screen,
            indicator_color,
            (WIDTH - 55, 45),
            12
        )

        notice = small_font.render(
            "SIMULATION ONLY - NOT FLIGHT COMMANDS",
            True,
            YELLOW
        )

        screen.blit(
            notice,
            (30, HEIGHT - 45)
        )

        pygame.display.flip()

    pygame.quit()