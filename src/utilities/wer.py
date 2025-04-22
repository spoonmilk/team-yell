import jiwer

# WER (Word Error Rate) calculation strategy taken from: https://medium.com/@johnidouglasmarangon/how-to-calculate-the-word-error-rate-in-python-ce0751a46052
# Thanks to John Douglas Marangon for the article and code snippet.


def wer(reference: str, hypothesis: str) -> float:
    """
    Calculate the Word Error Rate (WER) between a reference and a hypothesis.

    WER is calculated based on the equation WER = (S + D + I) / N, where:
    S = number of substitutions
    D = number of deletions
    I = number of insertions
    N = number of words in the reference

    Args:
        reference (str): Reference string/ground truth for the transcription.
        hypothesis (str): The hypothesis string/transcription to be evaluated.

    Returns:
        float: WER value
    """

    transforms = jiwer.Compose(
        [
            jiwer.ExpandCommonEnglishContractions(),
            jiwer.RemoveEmptyStrings(),
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords(),
        ]
    )

    wer = jiwer.wer(
        reference,
        hypothesis,
        truth_transform=transforms,
        hypothesis_transform=transforms,
    )
    return wer
