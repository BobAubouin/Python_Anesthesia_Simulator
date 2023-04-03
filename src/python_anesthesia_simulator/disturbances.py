import numpy as np


def compute_disturbances(time: float, dist_profil: str = 'realistic',
                         start_step: float = 600, end_step: float = 1200) -> list:
    """
    Give the value of the distubance profil for a given time.

    Parameters
    ----------
    time : float
        Time: in seconde.
    dist_profil : str, optional
        disturbance profil, can be: 'realistic', 'simple', 'step' or "null". The default is 'realistic'.
    start_step : float, optional
        start time of the step distuebance (seconds). The default is 600s.
    end_step : float, optional
        End time of the step distuebance (seconds). The default is 1200s.

    Returns
    -------
    list
        dist_bis, dist_map, dist_co: respectively the additive disturbance to add to the BIS, MAP and CO signals.

    """
    if dist_profil == 'realistic':
        # As proposed in M. M. R. F. Struys, T. De Smet, S. Greenwald, A. R. Absalom, S. Bingé, and E. P. Mortier,
        # “Performance Evaluation of Two Published Closed-loop Control Systems Using Bispectral Index Monitoring:
        #  A Simulation Study,”
        # Anesthesiology, vol. 100, no. 3, pp. 640–647, Mar. 2004, doi: 10.1097/00000542-200403000-00026.

        Disturb_point = np.array([[0,     0,  0, 0],  # time, BIS signal, MAP, CO signals
                                  [9.9,   0,  0, 0],
                                  [10,   20, 10, 0.6],
                                  [12,   20, 10, 0.6],
                                  [13,    0,  0, 0],
                                  [19.9,  0,  0, 0],
                                  [20.2, 20, 10, 0.5],
                                  [21,   20, 10, 0.5],
                                  [21.5,  0,  0, 0],
                                  [26,  -20, -10, -0.8],
                                  [27,   20, 10, 0.9],
                                  [28,   10,  7, 0.2],
                                  [36,   10,  7, 0.2],
                                  [37,   30, 15, 0.8],
                                  [37.5, 30, 15, 0.8],
                                  [38,   10,  5, 0.2],
                                  [41,   10,  5, 0.2],
                                  [41.5, 30, 10, 0.5],
                                  [42,   30, 10, 0.5],
                                  [43,   10,  5, 0.2],
                                  [47,   10,  5, 0.2],
                                  [47.5, 30, 10, 0.9],
                                  [50,   30,  8, 0.9],
                                  [51,   10,  5, 0.2],
                                  [56,   10,  5, 0.2],
                                  [56.5,  0,  0, 0]])
    elif dist_profil == 'simple':
        # As in G. A. Dumont, A. Martinez, and J. M. Ansermino,
        # “Robust control of depth of anesthesia,”
        # International Journal of Adaptive Control and Signal Processing,
        # vol. 23, no. 5, pp. 435–454, 2009, doi: 10.1002/acs.1087.

        Disturb_point = np.array([[0,     0,  0, 0],  # time, BIS signal, MAP, CO signals
                                  [19.9,  0,  0, 0],
                                  [20,   20,  5, 0.3],
                                  [23,   20, 10, 0.6],
                                  [24,   15, 10, 0.6],
                                  [26, 12.5,  6, 0.4],
                                  [30, 10.5,  4, 0.3],
                                  [37,   10,  4, 0.3],
                                  [40,    4,  2, 0.1],
                                  [45,  0.5, 0.1, 0.01],
                                  [50,    0,  0,   0]])
    elif dist_profil == 'step':
        Disturb_point = np.array([[0,     0,  0,   0],  # time, BIS signal, MAP, CO signals
                                  [start_step/60-0.01,   0,  0,   0],
                                  [start_step/60,    10,  5, 0.3],
                                  [end_step/60-0.01,   10,  5, 0.3],
                                  [end_step/60,  0,  0,   0],
                                  [30,    0,  0,   0]])

    elif dist_profil == 'null':
        return 0, 0, 0

    dist_bis = np.interp(time/60, Disturb_point[:, 0], Disturb_point[:, 1])
    dist_map = np.interp(time/60, Disturb_point[:, 0], Disturb_point[:, 2])
    dist_co = np.interp(time/60, Disturb_point[:, 0], Disturb_point[:, 3])

    return [dist_bis, dist_map, dist_co]
