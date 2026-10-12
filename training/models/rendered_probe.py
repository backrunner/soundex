"""P10 uses the unchanged P9 anchored amplitude network for all loss arms."""

from models.conditioned_shape import make_shape_model

ARMS = ("raw_control", "rendered_control", "rendered_balanced")


def make_rendered_model(arm, width=144):
    if arm not in ARMS:
        raise ValueError("unknown rendered study arm")
    return make_shape_model("anchored_band", width)
