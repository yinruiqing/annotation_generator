import numpy as np
from pyannote.core import Annotation, Segment, SlidingWindow
from pyannote.database.util import load_rttm

from utils import remove_non_speech, reset_start

class MimicAnnotationGenerator:
    """
    Parameters
    ----------
    rttm_file : `str`
        Path to rttm file
    keep_vad : bool, optional
        Keep non speech part or not. Defaults to 'False'.
    duration_noise : `float`, optional
        Defaults to 0.2 (20%).
    label_noise : `float`, optional
    length : `float`, in second
        Default to 6 seconds
    seed : `int`, optional
        Random seed.
    """

    def __init__(self, rttm_file, keep_vad=False, 
                 duration_noise=0.2, label_noise=0.05,
                 length=6, seed=None):

        self.duration_noise = duration_noise
        self.rttm_file = rttm_file
        self.keep_vad = keep_vad
        self.label_noise = label_noise
        self.length = length

        # set random seed
        np.random.seed(seed=seed)

        self.files_ = self._iter_files()
        

    def _iter_files(self):
        files = list(load_rttm(self.rttm_file).values())[:10]
        if not self.keep_vad:
            files = [remove_non_speech(f) for f in files]

        while True:
            np.random.shuffle(files)
            for current_file in files:
                yield current_file
        
    
    def random_crop_annotation(self, annotation):
        end = annotation.get_timeline().extent().end - self.length
        start = np.random.uniform(0, end)
        to_crop = Segment(start, start+self.length)
        return reset_start(annotation.crop(to_crop)).support()
            
        
    def __next__(self):
        annotation = next(self.files_)
        cropped_ann =  self.random_crop_annotation(annotation)
        if self.duration_noise == 0 and self.label_noise == 0:
            return cropped_ann.rename_labels(generator='int')
        else:
            res_ann = Annotation()
            K = len(cropped_ann.labels())
            for segment, _, k in cropped_ann.itertracks(yield_label=True):

                # randomly change segment label with probability p.
                if np.random.rand() > self.label_noise:
                    k = np.random.randint(K)

                # randomly shorten or lengthen segment
                # by up to "100 x self.duration_noise %""
                duration_noise = self.duration_noise * (2 * np.random.rand() - 1)
                duration = segment.duration * (1 + duration_noise)
                middle = segment.middle
                start = max(0, middle - duration / 2.0)
                end = min(self.length, middle + duration / 2.0)
                res_ann[Segment(start, end)] = k
            res_ann = reset_start(res_ann, start=0)
            res_ann = remove_non_speech(res_ann)
            end = res_ann.get_timeline().extent().end
            if end > self.length:
                return res_ann.crop(Segment(0, self.length)).support().rename_labels(generator='int')
            elif end < self.length: 
                res_ann[Segment(end, self.length)] = k
                return res_ann.support().rename_labels(generator='int')
            else:
                return res_ann.support().rename_labels(generator='int')