import numpy as np

NAN_COLS = ['b_out_x', 'b_out_y', 'b_spd', 'b_spd@+12', 'b_spd@+25', 'b_spd@+5', 'b_spd@+50', 'b_spd@-12',
            'b_spd@-25', 'b_spd@-5', 'b_spd@-50', 'b_x', 'b_y', 'd1_a', 'd1_a@+12', 'd1_a@+25', 'd1_a@+5',
            'd1_a@+50', 'd1_a@-12', 'd1_a@-25', 'd1_a@-5', 'd1_a@-50', 'd1_h', 'd1_h@+12', 'd1_h@+25', 'd1_h@+5',
            'd1_h@+50', 'd1_h@-12', 'd1_h@-25', 'd1_h@-5', 'd1_h@-50', 'd2_a', 'd2_h', 'd_diff', 'd_diff@+12',
            'd_diff@+25', 'd_diff@+5', 'd_diff@+50', 'd_diff@-12', 'd_diff@-25', 'd_diff@-5', 'd_diff@-50', 'b_acc']
CAP_COLS = ['since_ctrl_a', 'since_ctrl_h', 'since_touch_a', 'since_touch_h', 't_since_ball', 't_until_ball',
            'until_ctrl_a', 'until_ctrl_h', 'until_touch_a', 'until_touch_h']
ZERO_COLS = ['b_acc_max12', 'b_pres', 'b_pres_fut10', 'b_pres_fut100', 'b_pres_fut2000', 'b_pres_fut50',
             'b_pres_fut500', 'b_pres_m250', 'b_pres_past10', 'b_pres_past100', 'b_pres_past2000', 'b_pres_past50',
             'b_pres_past500', 'b_spd_max25', 'b_still', 'ctrl_frac_a_250', 'ctrl_frac_a_50', 'ctrl_frac_h_250',
             'ctrl_frac_h_50', 'last_touch_diff', 'n10_a', 'n10_h', 'n3_a', 'n3_h', 'next_touch_diff', 'touch_a',
             'touch_chain', 'touch_h', 'touch_rate_a_250', 'touch_rate_h_250']
CAP = 3000.0


def _idx(names):
    names = list(names)
    return ([names.index(c) for c in NAN_COLS], [names.index(c) for c in CAP_COLS],
            [names.index(c) for c in ZERO_COLS])


def blind_rows(X, names):
    a, b, c = _idx(names)
    Y = np.array(X, dtype=np.float32, copy=True)
    Y[:, a] = np.nan
    Y[:, b] = CAP
    Y[:, c] = 0.0
    return Y


def blind_pooled(P, names, n_ind=6):
    F = len(names)
    if P.shape[1] != F + n_ind:
        raise ValueError("pooled view has %d columns, expected %d + %d" % (P.shape[1], F, n_ind))
    Y = P.copy()
    Y[:, :F] = blind_rows(P[:, :F], names)
    Y[:, F:] = 1.0
    return Y


def check_against(X_masked, names, tol=1e-3):
    a, b, c = _idx(names)
    X = np.asarray(X_masked)
    bad = (np.mean(~np.isnan(X[:, a])) > tol or np.mean(X[:, b] != CAP) > tol or np.mean(X[:, c] != 0) > tol)
    if bad:
        raise RuntimeError("blind template does not match the masked-ball view (featset changed?)")
