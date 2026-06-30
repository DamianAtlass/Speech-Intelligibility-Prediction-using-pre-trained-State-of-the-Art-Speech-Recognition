################################################################################
#
# Miniproject 2026 - Berlin
# 23.03.2026 - 27.03.2026
#
# Alexander Wißmann, Jasmin Riegel, Steffen Zeiler, Dorothea Kolossa
# Technische Universität Berlin
# Friedrich-Alexander Universität Erlangen-Nürnberg
#
################################################################################
import numpy as np


def _needlemann_wunsch(reference, transcript):
    """
    Dynamic programming algorithm to align true transcription with output sequence in order to find the smallest distance.

    :param reference: Reference sequence (true transcription).
    :param transcript: output of viterbi

    :return ref_align: Alignment for reference.
    :return trans_align: Alignment for transcript.
    """
    gap_score = -1
    sim_func = lambda x, y: 0 if x == y else -1

    n_ref = len(reference)
    n_trans = len(transcript)
    d_mat = np.zeros(shape=[n_trans + 1, n_ref + 1, ])
    # Initialize the dynamic programming calculation using base conditions
    for idr in range(n_ref + 1):
        d_mat[0, idr] = gap_score * idr
    for idt in range(n_trans + 1):
        d_mat[idt, 0] = gap_score * idt

    # Calculate all D[i,j]
    for i in range(1, n_trans + 1):
        for j in range(1, n_ref + 1):
            match = d_mat[i - 1, j - 1] + sim_func(reference[j - 1], transcript[i - 1])
            gaps = d_mat[i, j - 1] + gap_score
            gapt = d_mat[i - 1, j] + gap_score
            d_mat[i, j] = np.max([match, gaps, gapt])

    # Do the Traceback to create the alignment
    i = n_trans
    j = n_ref
    ref_align = []
    trans_align = []
    while (i > 0) and (j > 0):
        if d_mat[i, j] - sim_func(reference[j - 1], transcript[i - 1]) == d_mat[i - 1, j - 1]:
            ref_align.insert(0, reference[j - 1])
            trans_align.insert(0, transcript[i - 1])
            i = i - 1
            j = j - 1
        elif d_mat[i, j] - gap_score == d_mat[i, j - 1]:
            ref_align.insert(0, reference[j - 1])
            trans_align.insert(0, None)
            j = j - 1
        elif d_mat[i, j] - gap_score == d_mat[i - 1, j]:
            ref_align.insert(0, None)
            trans_align.insert(0, transcript[i - 1])
            i = i - 1
        else:
            raise ('should not happen')

    while j > 0:
        ref_align.insert(0, reference[j - 1])
        trans_align.insert(0, None)
        j = j - 1

    while i > 0:
        ref_align.insert(0, None)
        trans_align.insert(0, transcript[i - 1])
        i = i - 1

    return (ref_align, trans_align)


def needlemann_wunsch(reference: str, transcript: str):
    """
    Counts number of errors.

    :param reference: Reference sequence (true transcription).
    :param transcript: output of viterbi

    :return N: Total number of words.
    :return D: Number of deleted words.
    :return I: Number of inserted words.
    :return S: Number of substituted words.
    """
    insertions = 0
    deletions = 0
    substitutions = 0
    ref_align, trans_align = _needlemann_wunsch(reference, transcript)
    for idr in range(len(ref_align)):
        if ref_align[idr] is None:
            insertions += 1
        elif trans_align[idr] is None:
            deletions += 1
        elif ref_align[idr] != trans_align[idr]:
            substitutions += 1

    return len(reference), deletions, insertions, substitutions

######## own additions ############

def wer_needleman_wunsch(reference: list, transcript: list) -> float:
    total_elements, deletions, insertions, substitutions = 0, 0, 0, 0

    for r, t in zip(reference, transcript):
        t, d, i, s = needlemann_wunsch(reference=r,transcript=t)
        total_elements += t
        deletions += d
        insertions += i
        substitutions += s

    return (substitutions + insertions + deletions) / total_elements

if __name__ == '__main__':
    total, deletions, insertions, substitutions = needlemann_wunsch(reference="test test test", transcript="test besti test")

    print((substitutions + insertions + deletions)/total)

    long_ref = "red blue six"
    long_hypothesis = "red blue sicks"
    import werpy
    print(werpy.wer(long_ref, long_hypothesis))
    print(wer_needleman_wunsch(reference=[long_ref], transcript=[long_hypothesis]), )