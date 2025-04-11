


# dual log version
import matplotlib
from matplotlib.colors import LinearSegmentedColormap

def dual_log_cmaps():
    cdict3 = {'red':   [(0.0,  1.0, 1.0), # white
                        (0.33,  1.0, 1.0), # orange
                        (0.66,  1.0, 1.0), # pink
                        (1.0,  1.0, 1.0)], # red 

            'green': [(0.0,  0.0, 1.0),
                        (0.33, 170/255.0, 170/255.0), # (0.33, 170.0/255.0, 170.0/255.0),
                        (0.66, 0.0, 0.0),
                        (1.0,  0.0, 0.0)],

            'blue':  [(0.0,  0.0, 1.0),
                        (0.33,  0.0, 0.0),
                        (0.66,  1.0, 1.0),
                        (1.0,  0.0, 0.0)]}


    cmap1 = LinearSegmentedColormap('WhiteYellowPinkRed',cdict3)
    cmap1.set_bad('white',alpha=0.0)

                
    cdict4 = {'red':   [(0.0,  1.0, 1.0),
                        (0.33,  0.0, 0.0),
                        (0.66,  0.0, 0.0),
                        (1.0,  0.0, 0.0)],

            'green': [(0.0,  0.0, 1.0),
                        (0.33, 1.0, 1.0),
                        (0.66,  1.0, 1.0),
                        (1.0,  0.0, 0.0)],

            'blue':  [(0.0,  0.0, 1.0),
                    (0.33,  0.0, 0.0),
                    (0.66,  1.0, 1.0),
                    (1.0,  1.0, 1.0)]}

    cmap2 = LinearSegmentedColormap('WhiteGreenCyanBlue',cdict4)
    cmap2.set_bad('white',alpha=0.0)

    cdict5 = {'red':   [(0.0,  0.0, 0.0),
                                    (0.16,  0.0, 0.0),
                                    (0.33,  0.0, 0.0),
                                    (0.5,  1.0, 1.0), # white
                                    (0.66,  1.0, 1.0), # orange
                                    (0.83,  1.0, 1.0), # pink
                                    (1.0,  1.0, 1.0)], # red 

            'green': [(0.0,  0.0, 0.0),
                    (0.16,  1.0, 1.0),
                    (0.33,  1.0, 1.0),
                    (0.5,  1.0, 1.0),
                    (0.66, 170.0/255.0, 170.0/255.0), # (0.33, 170.0/255.0, 170.0/255.0),
                    (0.83, 0.0, 0.0),
                    (1.0,  0.0, 0.0)],

            'blue':  [(0.0,  1.0, 1.0),
                    (0.16,  1.0, 1.0),
                    (0.33,  0.0, 0.0),
                    (0.5,  1.0, 1.0),
                    (0.66,  0.0, 0.0),
                    (0.83,  1.0, 1.0),
                    (1.0,  0.0, 0.0)]}
                # for the colobars only
    cmap3 = LinearSegmentedColormap('BlueCyanGreenWhiteYellowPinkRed',cdict5)
    cmap3.set_bad('white',alpha=0.0)
    return cmap1,cmap2, cmap3

