def align_seqs(pred: list, actual: list, pad="<?UNK?>"):
    """
    Align two sequences (`pred` and `actual`) by inserting `pad`ding elements into the shorter sequence such that we
    maximise the number of matches between the two lists. Elements are never swapped in order or transmuted.

    Examples
    ========

    A shorter predicted sequence will have a padding element inserted:
    >>> print(align_seqs(["Hey", "there", "neighbour!"], ["Hey", "there", "my", "neighbour!"]))
        (['Hey', 'there', '<?UNK?>', 'neighbour!'], ['Hey', 'there', 'my', 'neighbour!'])

    A shorter actual sequence will have a padding element inserted:
    >>> print(align_seqs(["Hey", "there", "neighbour!"], ["Hey", "there"]))
        (['Hey', 'there', 'neighbour!'], ['Hey', 'there', '<?UNK?>'])

    Sequences of equal length do not change:
    >>> print(align_seqs(["Hey", "hello", "there", "neighbour!"], ["Hey", "there", "my", "neighbour!"]))
        (['Hey', 'hello', 'there', 'neighbour!'], ['Hey', 'there', 'my', 'neighbour!'])
    """
    longer, shorter = (pred, actual) if len(pred) > len(actual) else (actual, pred)

    best_correct = 0
    best_indices = []

    for indices in _gen_indices(len(longer) - len(shorter), len(longer) - 1):
        # Just copy the shorter one as it's what we will insert into
        pred_copy, actual_copy = (pred, _copy_and_insert(actual, indices, pad)) if len(pred) > len(actual) else (
            _copy_and_insert(pred, indices, pad), actual)

        correct = 0
        for (pred_elt, actual_elt) in zip(pred_copy, actual_copy):
            if pred_elt == actual_elt:
                correct += 1
    
        if correct >= best_correct:
            best_indices = indices
            best_correct = correct

        if best_correct == len(shorter):
            break

    shorter = _copy_and_insert(shorter, best_indices, pad)

    return (longer, shorter) if len(pred) > len(actual) else (shorter, longer)

def _copy_and_insert(seq: list, indices, pad):
    """Copy the given `seq` and insert the `pad` element at the specified `indices`, returning the post-insert list"""
    copied = seq.copy()

    for index in indices:
        copied.insert(index, pad)

    return copied


def _gen_indices(length_diff: int, max_idx: int) -> list:
    """
    Generate all possible sets of indices that we can insert elements into the smaller list in order for it to
    become as long as the longest list. In this function, we are not concerned with which are the _optimal_ indices -
    we just generate all of them and filter later.
    """

    # Let's say we have shortest_len = 7 and longest = 10
    # We need to find (x, y, z) s.t. x <= y <= z
    # So, first pick x in the range 10..=0
    # Then we pick y in the range x..=0
    # Then z in the range y..=0
    # etc

    if length_diff == 0:
        return []

    for first_coord in range(max_idx, -1, -1):
        if length_diff == 1:
            yield (first_coord,)
        else:
            for indices in _gen_indices(length_diff - 1, first_coord):
                yield first_coord, *indices
