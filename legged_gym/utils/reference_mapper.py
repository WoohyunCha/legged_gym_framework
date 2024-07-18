import numpy as np

class reference_mapper():
    def __init__(self, reference_motion: np.ndarray, source_info: dict) -> None:
        self.reference_motion = reference_motion
        self.source_info = source_info
        print("Reference mapper initialized. Source info and reference motion are given")
        
    '''
    Maps the reference motion to target configurations.
    '''
    def retarget(self, target_info: dict) -> np.ndarray:

        retarget_result = None #TODO
        return retarget_result