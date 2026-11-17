import numpy as np

from vision.refine.model import cell_centres
from vision.refine.sample import viterbi
from vision.refine.jumps import frame_off, despike, cell_interp

LAM = 30.0
CI_LEN = 250


def final_xy(res, cand, lam=LAM, ci_len=CI_LEN):
    centres = cell_centres().numpy()
    xv, vt = viterbi(cand, res["top_l"], res["top_i"], frame_off(res, centres), centres, lam=lam)
    if ci_len:
        xv = cell_interp(xv, vt, ci_len)
    return despike(xv).astype(np.float32), vt
