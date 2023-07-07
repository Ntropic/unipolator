# -*- coding: utf-8 -*-
from numpy import diff, sqrt, min, linspace, arange, NaN
from scipy.special import erfinv
import os
from matplotlib import pyplot as plt
from tikzplotlib import get_tikz_code
from Analysis_Code.useful import *

def tex_exists_and_same(filename, code):
    # check if .tex exists and if it is the same as the code
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            old_code = f.read()
        # ignore comments in code %, empty lines and trailing and leading spaces
        # separate both codes into lines and compare them after removing empty lines, comments and trailing and leading spaces
        code_lines = [line.strip() for line in code.split('\n')]
        old_code_lines = [line.strip() for line in old_code.split('\n')]
        # remove comments
        code_lines = [line.split('%')[0].strip() for line in code_lines]
        old_code_lines = [line.split('%')[0].strip() for line in old_code_lines]
        # remove empty lines
        code_lines = [line for line in code_lines if line]
        old_code_lines = [line for line in old_code_lines if line]
        # compare line by line
        figures_new = 0
        if len(code_lines) == len(old_code_lines):
            for i, (line, old_line) in enumerate(zip(code_lines, old_code_lines)):
                if line != old_line:
                    if line != old_line:
                        # check if difference is a number, followed by".png"   --> 
                        for j, (char, old_char) in enumerate(zip(line, old_line)):
                            if char != old_char:
                                break
                        # check if change is a number
                        if line[j].isdigit():
                            while line[j].isdigit():
                                j += 1
                                if j >= len(line)-1:
                                    break
                            if j+4 < len(line):
                                if '.png' == line[j:j+4]:
                                    figures_new = 1
                                    pass   # figure the same but different name (numbering) 
                                else:
                                    print('Figure changed, updating file. ', end=' ')
                                    return -1
                            else:
                                print('Figure changed, updating file. ', end=' ')
                                return -1
                        else:
                            print('Figure changed, updating file. ', end=' ')
                            return -1
            print('Figure unchanged, not updating file.')
            if figures_new:
                return 2
            else:
                return 1
        else:
            print('Figure changed, updating file. ', end=' ')
            return -1
    print('File ('+filename+') does not exist, creating file. ', end=' ')
    return 0
    
def fig2tikz(fig, filename, axis_width=8.0, axis_height=6.375, dir='Tikz', add_letter='', legend_columns=2, dont_check_existing=0):
    full_filename = os.path.join(dir, filename)
    if not full_filename.endswith('.tex'):
        full_filename += '.tex'
    name_without_ending = filename
    if name_without_ending.endswith('.tex'):
        name_without_ending = name_without_ending[:-4]
    # make a list of files in the directory and compare if new files have been added
    files_in_directory = os.listdir(dir)
    code = get_tikz_code(figure=fig, filepath=full_filename, axis_width=str(axis_width)+'cm', axis_height=str(axis_height)+'cm')
    new_files_in_directory = os.listdir(dir)
    added_files = []
    for file in new_files_in_directory:
        if name_without_ending in file:
            if file not in files_in_directory:
                added_files.append(file)
    #### modify the string
    # add start and end strings to code  #legend columns=2,
    code_list = code.split('\n')
    ind = -1
    for i, line in enumerate(code_list):
        if line.startswith(r'\begin{axis}'):
            ind = i+1
            break
    if ind > -1:
        code_list = code_list[:ind] + [r'axis on top=true,', 'clip marker paths=true,'] + code_list[ind:]
    if len(add_letter):
        # find the line in code_list that starts with  \end{axis}
        ind = -1
        for i, line in enumerate(code_list):
            if line.startswith(r'\end{axis}'):
                ind = i
                break
        if ind > -1: 
            x_pos = 0.02
            y_pos = 0.97
            new_line = r'\node[scale=1.0, anchor=north west, text=black,  rotate=0.0]  at (rel axis cs:' + str(x_pos) + ',' + str(y_pos) + ') {\contour{white}{(' + add_letter + ')}};'
            code_list = code_list[:ind] + [new_line] + code_list[ind:]
            
    #append to beggining of code_list
    code_list = [r'\documentclass[groupedaddress,amsmath,amssymb,amsfonts,nofootinbib,a4paper, 10pt]{standalone}', r'\input{../tikz_header}', r'\begin{document}'] + code_list + [r'\end{document}']
    # first make shure that legend is even used
    # check if any lines start with "legend "
    has_legend = False
    for line in code_list:
        if line.strip().startswith('legend '):
            has_legend = True
            break
    if has_legend:
        code_list = replace_or_add_between_in_brace_block(code_list,'legend style={', 'at={(', ')}', '0.5, 1.05', addition=', at={(0.5,1.05)}')
        # find the brace block that starts with \legend style={ and ends with }
        start_line, start_ind, end_line, end_ind = find_brace_block(code_list, 'legend style={')
        code_list[end_line] = code_list[end_line][:end_ind] + r', legend columns='+str(legend_columns) + ', anchor=south' + code_list[end_line][end_ind:]
    code = '\n'.join(code_list)
    # save code to file
    # check if .tex exists and if it is the same as the code
    if dont_check_existing == 0:
        exists_and_same = tex_exists_and_same(full_filename, code) # 0: does not exist, 1: exists and same, -1: exists but different figures --> delete files first
    else:
        exists_and_same = 0
    if exists_and_same == 0:
        with open(full_filename, 'w') as f:
            f.write(code)
        print('File saved to: '+full_filename)
    elif exists_and_same == -1:   # 
        # delete all files with the name in them using name_without_ending ans os.remove
        for file in os.listdir(dir):
            if name_without_ending in file:
                os.remove(os.path.join(dir, file))
        # call script fig2tikz again to save file 
        fig2tikz(fig, filename, axis_width=axis_width, axis_height=axis_height, dir=dir, add_letter=add_letter, legend_columns=legend_columns, dont_check_existing=1)
    else:
        for file in added_files:
            os.remove(os.path.join(dir, file))
        
# a function that extracts the axis environments from multiple .tex files with tikz graphics described.
# the script then adds the axis into a new file with the header and footer from the first of the original files.
# furthermore the legend is extracted from the first file and added to the new file, but removed from the others.
# the legend is then centered above the figures
def extract_axis_from_tex(filename):
    # load the file with the filename, separate the header, footer and axis environments
    with open(filename, 'r') as f:
        lines = f.readlines()
    # find the first line that has begin{axis}, and the last line with end{axis}
    begin = None
    end = None
    for i, line in enumerate(lines):
        if '\\begin{axis}' in line and begin is None:
            begin = i
        if '\\end{axis}' in line:
            end = i
    # extract the header and footer
    header = lines[:begin]
    footer = lines[end+1:]
    # extract the axis
    axis = lines[begin:end+1]
    # extract the legend
    # find the (lines, that are connected to the legend) (startwtih 'legend ')
    legend_lines = [i for i, line in enumerate(axis) if line.strip().startswith('legend ')]
    return header, footer, axis, legend_lines

# a script to load multiple files froma file_list and extract the axis environments from them, then combining them into a new file
def combine_tex_files(file_list, output_filename, columns=2, redo=False, directory=None, shift_by=1.0, shift_by_x=None, shift_by_y=None, x_shift=None, y_shift=None, columns_half=None): #'Tikz'
    how_many = len(file_list)
    if not directory is None:
        file_list = [os.path.join(directory, filename) for filename in file_list]
        output_filename = os.path.join(directory, output_filename)
    if how_many % columns != 0:
        print('The number of files is not divisible by the number of columns')
        return
    # check if filenames end with .tex if not add it
    for i, filename in enumerate(file_list):
        if not filename.endswith('.tex'):
            file_list[i] = filename+'.tex'
    if not output_filename.endswith('.tex'):
        output_filename += '.tex'
    # check if the file_list exists
    for filename in file_list:
        if not os.path.exists(filename):
            print('The file: '+filename+' does not exist')
            return
    # check if the output_filename exists
    if not redo:
        if os.path.exists(output_filename):
            # is it newer than the other files?
            output_time = os.path.getmtime(output_filename)
            #check if any of the files have been updated since the output file was created
            for filename in file_list:
                if os.path.getmtime(filename) > output_time:
                    # redo the file
                    redo = True 
                    print('Updating the file: '+output_filename+' because the file: '+filename+' has been updated')
                    break
            print('The file: '+output_filename+' is up to date')
        else:
            print('The file: '+output_filename+' does not exist')
            redo = True
    if redo:
        headers, footers, axises, legend_lines = [], [], [], []
        for i, filename in enumerate(file_list):
            header, footer, axis, legend_line = extract_axis_from_tex(filename)
            # append to lists
            headers.append(header)
            footers.append(footer)
            axises.append(axis)
            legend_lines.append(legend_line)
        #determine width and height of the figures from the first figure
        axis = axises[0]
        # find the width and height of the figure (line with width= and height=)
        width = -1
        height = -1
        for line in axis:
            if 'width=' in line:
                width = float(line.split('=')[1].split('cm')[0])
            if 'height=' in line:
                height = float(line.split('=')[1].split('cm')[0])
            if width != -1 and height != -1:
                break
        #width = 8.5   # determine from files
        #height= 6.375 # determine from files
        if shift_by_x is None:
            shift_by_x = shift_by
        if shift_by_y is None:
            shift_by_y = shift_by
        if x_shift is None:
            x_shift = width + shift_by_x
        if y_shift is None:
            y_shift = height + shift_by_y
        if columns_half is None:
            columns_half = columns/2*x_shift/width
        str_half = str(columns_half)
        # combine the headers and footers
        new_document = headers[0]
        new_document.append(r'\begin{scope}'+'\n')
        for i, (header, footer, axis, legend_line) in enumerate(zip(headers, footers, axises, legend_lines)):
            if i == 0:
                # combine the axises
                # change legend style --> at={(column_half,1.05)}
                # find that line
                for j in legend_line:
                    line = axis[j]
                    if 'legend style={' in line:
                        # find the number after style={at={(, and before the ,
                        # use the function: replace_between(string, first, last, new)
                        # find all the lines belonging to the block
                        start_line, _, end_line, _ = find_brace_block(axis[j:], 'legend style={')
                        axis[j+start_line:j+end_line+1] = replace_or_add_between_in_brace_block(axis[j+start_line:j+end_line+1],'legend style={', 'at={(', ',', str_half, addition=', at={('+str_half+',1.05)}}')
                new_document += axis
            else:
                # add delimiter between the figures
                which_column = i % columns
                which_row = i // columns
                new_document.append(r'\begin{scope}[xshift='+str(x_shift*which_column)+'cm, yshift=-'+str(y_shift*which_row)+'cm]'+'\n')
                # comment out the current legend lines
                for j in legend_line:
                    if '={' in axis[j]:
                        # find the start and end of the block
                        start_line, _, end_line, _ = find_brace_block(axis[j:], '={')
                        for k in range(start_line+j, end_line+1+j): 
                            axis[k] = '%'+axis[k]
                    else: 
                        axis[j] = '%'+axis[j]
                for j in range(len(axis)):
                    # remove addlegendentry line
                    if 'addlegendentry' in axis[j]:
                        axis[j] = '%'+axis[j]
                # combine the axises
                new_document += axis
            new_document.append(r'\end{scope}'+'\n')
        # add the footers
        new_document += footers[0]
        # write the new document to a file
        with open(output_filename, 'w') as f:
            # write the new document
            print('Writing to file: '+output_filename)
            f.writelines(new_document)
        
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

def min_mean_max_plot(axes, x, y, miny, maxy, ps=0.68, label='', alphas=None, colors=None, legendary=[], support=[], logy=0, **kwargs):
    """
    Draw a plot and fill the area around it using min and max vals.
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
    alpha_max = 0.75
    d = erfinv(linspace(0, ps, 2)) * sqrt(2)
    differ = float(diff(d))
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

    if isinstance(y, list) == 0:
        x = [x]
        y = [y]
        miny = [miny]
        maxy = [maxy]
    if isinstance(x, list) == 0:
        n = len(y)
        x = [x]*n
    for i, (cx, cy, mini, maxi) in enumerate(zip(x, y, miny, maxy)):
        if colors is not None:
            if not isinstance(colors, str):
                axes.set_prop_cycle(color=colors)
                color = axes._get_lines.get_next_color()
            else:
                color = colors
        else:
            color = axes._get_lines.get_next_color()
        if len(support):
            cy = map_to_support(cy, support).astype(float)
            mini = map_to_support(mini, support).astype(float)
            maxi = map_to_support(maxi, support).astype(float)
        if len(cx) < len(cy):
            cx = arange(len(cy))
        ###### Plot std shadings
        p2 = axes.fill_between(cx, mini, maxi, color=color, alpha=alphas, edgecolor='none', **kwargs)

        ##### Plot expectation value
        if len(cx) > 1:
            if logy:
                p1 = axes.semilogy(cx, cy, color=color, **kwargs)
            else:
                p1 = axes.plot(cx, cy, color=color, **kwargs)

            if i == 0:
                p11 = axes.fill(NaN, NaN, color=color, alpha=alphas)
                r.append(p1)
                r.append(p2)
                if label:
                    handles = [*prev_handles, (p1[0], p11[0])]
                    labels = [*prev_labels, label]
                else:
                    handles = [*prev_handles]
                    labels = [*prev_labels]
                        
        else:
            handles = [*prev_handles]
            labels = [*prev_labels]
            mini = 1
    return [handles, labels], mini


# a function that finds width= and height= in a file and replaces the values, if specified, 
# if square is specified then if width and height are the same they remain the same 
# after the transformation
def resize_tikz(filename, width=None, height=None, square=False, directory=None):
    if width is None and height is None:
        raise ValueError("Must specify at least one of width or height")
    if not directory is None:
        filename = os.path.join(directory, filename)
    # does file exist?
    if not os.path.isfile(filename):
        raise ValueError("File does not exist: {}".format(filename))
    with open(filename, 'r') as f:
        lines = f.readlines()
    # find width and height inside of the file
    width_index = -1
    height_index = -1
    old_width, old_height = None, None
    for i, line in enumerate(lines):
        if line.startswith('%'):
            continue
        if 'width=' in line:
            old_width = float(line.split('=')[1].split('cm')[0])
            width_index = i
        if 'height=' in line:
            old_height = float(line.split('=')[1].split('cm')[0])
            height_index = i
    
    if width_index == -1:
        print("No width found in file: {}".format(filename))
    if height_index == -1:
        print("No height found in file: {}".format(filename))
    if old_width == old_height and square:
        if not width is None:
            height = width
        else:
            width = height
    # change the width and height
    if not width is None and not width_index == -1:
        lines[width_index] = lines[width_index].split('=')[0] + '=' + str(width) + 'cm,\n'
    if not height is None and not height_index == -1:
        lines[height_index] = lines[height_index].split('=')[0] + '=' + str(height) + 'cm,\n'
    # write the file
    with open(filename, 'w') as f:
        f.writelines(lines)
        print("Updated file: {}".format(filename))
        
def resize_all_subdirectories(directory=None, width=None, height=None, square=False, depth=1):
    # Navigate through all subdirectories and resize all tikz files
    # navigate through at most depth subdirectories
    if directory is None:
        directory = os.getcwd()
    for root, dirs, files in os.walk(directory):
        if root[len(directory):].count(os.sep) < depth:
            for file in files:
                if file.endswith(".tex"):
                    resize_tikz(file, width=width, height=height, square=square, directory=root)
                    
        