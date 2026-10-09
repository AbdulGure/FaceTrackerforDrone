def associate_face_to_person(face_box, people):
    """
    Match a recognised face to a detected person.

    Returns the matching person's bounding box,
    or None if the association is ambiguous.
    """

    fx1, fy1, fx2, fy2 = face_box

    face_center_x = (fx1 + fx2) / 2
    face_center_y = (fy1 + fy2) / 2

    candidates = []

    for person in people:

        x1, y1, x2, y2 = person["bbox"]

        width = x2 - x1
        height = y2 - y1

        if width <= 0 or height <= 0:
            continue

        # The face should be horizontally inside
        # the person bounding box and approximately
        # within the upper half of the body.
        inside_x = x1 <= face_center_x <= x2

        inside_y = (
            y1 <= face_center_y <=
            y1 + height * 0.5
        )

        if inside_x and inside_y:
            candidates.append(person)

    # Reject ambiguous associations rather than
    # selecting an arbitrary person.
    if len(candidates) != 1:
        return None

    return candidates[0]["bbox"]