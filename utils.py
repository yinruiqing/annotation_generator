from pyannote.core import Annotation, Segment, Timeline

def remove_non_speech(annotation):
    ann_start = annotation.get_timeline()[0].start
    non_speech = annotation.get_timeline().gaps()
    if ann_start != 0:
        non_speech.add(Segment(0, ann_start))
    result_annotation = Annotation(annotation.uri)
    for segment, _, label in annotation.itertracks(yield_label=True):
        non_speech_duration = non_speech.crop(Segment(0, segment.end)).duration()
        new_segment = Segment(segment.start-non_speech_duration, segment.end-non_speech_duration)
        result_annotation[new_segment] = label
    return result_annotation


def reset_start(annotation, start=0):
    original_start = annotation.get_timeline()[0].start
    diff = original_start - start
    result_annotation = Annotation(annotation.uri)
    for segment, _, label in annotation.itertracks(yield_label=True):
        new_segment = Segment(segment.start-diff, segment.end-diff)
        result_annotation[new_segment] = label
    return result_annotation