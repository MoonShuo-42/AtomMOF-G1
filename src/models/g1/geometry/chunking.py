"""Bound temporary geometry work by both atom pairs and periodic image slots."""


def geometry_chunk_size(pair_limit: int, image_limit: int, image_slots: int) -> int:
    """No device reads: slot count was already computed once by the search plan.

    image_limit bounds padded pair-image entries, not the accepted neighbor count.
    It never discards a pair/image. A single pair is indivisible.
    """
    return min(pair_limit, max(1, image_limit // max(1, image_slots)))
