# -*- coding: utf-8 -*-
from numpy import diff, sqrt, min, linspace, arange, NaN
from scipy.special import erfinv

def map_to_support(vals, support, none_ind=1):
    # A function that maps values onto a support - None values are mapped to one of the extremes of support via the none_index (default =1)
    # first map none values
    if support[1] < support[0]:
        support[0], support[1] = support[1], support[0]
    vals[vals == None] = support[none_ind]
    vals[vals > support[1]] = support[1]
    vals[vals < support[0]] = support[0]
    return vals

def steps_xy(x,y):
    fx = [item for item in x[:-1] for i in range(2)] + [x[-1]]
    fy = [y[0]] + [item for item in y[1:] for i in range(2)] 
    return fx, fy

##### Plot Types #######################################################################################################

def std_plot(axes, x, y, std, ps=0.9, ns=1, label='', alphas=None, colors=None, legendary=[], support=[], logy=0, **kwargs):
    """
    Draw a plot and fill the standard deviation area around it .
    Parameters
    ----------
    axes : the axes object on which to draw
    x : (N,) array-like or list of arrays
    y : (N,) array-like or list of arrays
    ds : step width in multiples of standard deviation
    ns : number of steps to plot
    label : str, optional to specify label in legend
    colors : colors, optional
    legendary : legend [handles, labels], optional => get via legendary=list(axes.get_legend_handles_labels())
    support : optional, specifies the y support [min max]
    logy : optional (defualt 0) semilogy axis
    **kwargs
        All other keyword arguments are passed to `.Axes.fill_between` and `.Axes.plot`.
    Returns
    -------
    List of `.PolyCollection`
    Updated legendary list of [handles, labels]
    -------
    Add legend via axes.legend(*legendary)
    """
    if ns > 0:
        alpha_max = 0.75
        d = erfinv(linspace(0, ps, ns+1)) * sqrt(2)
        differ = diff(d)
        if alphas == None:
            alphas = alpha_max * min(differ) / differ

    if len(legendary):
        prev_handles = legendary[0]
        prev_labels = legendary[1]
    else:  # Add: list(axes.get_legend_handles_labels())
        prev_handles=[]
        prev_labels=[]

    r = []
    # Color between array i-1 and array i

    if not isinstance(y, list):
        x = [x]
        y = [y]
        std = [std]
    elif not isinstance(x, list):
        n = len(y)
        x = [x]*n
    for i, (cx, cy, cstd) in enumerate(zip(x, y, std)):
        if colors is not None:
            if not isinstance(colors, str):
                axes.set_prop_cycle(color=colors)
                color = axes._get_lines.get_next_color()
            else:
                color = colors
        else:
            color = axes._get_lines.get_next_color()
        if len(support):
            cy[cy > support[1]] = support[1]
            cy[cy < support[0]] = support[0]
        if len(cx) < len(cy):
            cx = arange(len(cy))
        ###### Plot std shadings
        how_many = 1
        for n in range(ns):
            for j in range(how_many):
                if n > 0:
                    sign = -1 if j else +1
                    upper_min = cy + sign * d[n] * cstd
                    upper_max = cy + sign * d[n+1 ] * cstd
                else:
                    #print(cy[-1], d[n+1 ], cstd[-1])
                    upper_min = cy -  d[n+1 ] * cstd
                    upper_max = cy +  d[n+1 ] * cstd
                if len(support):
                    upper_min[upper_min < support[0]] = support[0]
                    upper_max[upper_max < support[0]] = support[0]
                    upper_min[upper_min > support[1]] = support[1]
                    upper_max[upper_max > support[1]] = support[1]
                if n == ns-1:
                    mini_now = min(upper_min)
                    if i > 0:
                        if mini > mini_now:
                            mini = mini_now
                    else:
                        mini = mini_now
                if len(cx) > 1:
                    p2 = axes.fill_between(cx, upper_min, upper_max, color=color, alpha= alphas[n], edgecolor='none', **kwargs)
            how_many = 2

        ##### Plot expectation value
        if len(cx) > 1:
            if logy:
                p1 = axes.semilogy(cx, cy, color=color, **kwargs)
            else:
                p1 = axes.plot(cx, cy, color=color, **kwargs)

            if ns > 0:
                if i == 0:
                    p11 = axes.fill(NaN, NaN, color=color, alpha=alphas[0])
                    r.append(p1)
                    r.append(p2)
                    if label:
                        handles = [*prev_handles, (p1[0], p11[0])]
                        labels = [*prev_labels, label]
                    else:
                        handles = [*prev_handles]
                        labels = [*prev_labels]
            else:
                mini = min(cy)
                if i == 0:
                    r.append(p1)
                    if label:
                        handles = [*prev_handles, p1[0]]
                        labels = [*prev_labels, label]
                    else:
                        handles = [*prev_handles]
                        labels = [*prev_labels]
        else:
            handles = [*prev_handles]
            labels = [*prev_labels]
            mini = 1
    return [handles, labels], mini

