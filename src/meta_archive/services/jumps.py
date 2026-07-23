from __future__ import annotations


def activity_jump(activity_id: str) -> dict[str, str]:
    """Return the canonical, stable browser target for one retained activity record."""

    fragment = f"activity-{activity_id}"
    return {
        "href": f"/timeline?around=activity:{activity_id}#{fragment}",
        "fragment": fragment,
        "activity_id": activity_id,
    }
