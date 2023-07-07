import os
import shutil
from numpy import floor, log10, abs
import numpy as np
from itertools import product
from tqdm import tqdm

#### Lines #####################################################################
def line(title='', left=5, right=-1, line_style='—'):
    # This function prints a line with a title in it
    if right < 0:
        try:
            width, _ = os.get_terminal_size()
        except:
            terminal_size = shutil.get_terminal_size()
            width = terminal_size.columns
        right = width - left
    if len(title):
        title = ' '+title+' '
        len_title = len(title)
        right = right-len_title
    if right < 0:
        right = 0
    print(line_style*left+title+line_style*right,end='\n')
def dline(title='', left=5, right=-1):
    # This function prints a double line with a title in it
    line(title, left, right, line_style='═')
    
#### Numbers ###################################################################
def fstr(number, precision = 3):
    # This function returns a string of a number with a given precision x.xxxx
    num = float(number)
    whole = int(floor(num))
    fraction = int(round((num-whole)*10**precision))
    str1 = str(fraction)
    str_fraction = "0"*(precision-len(str1))+str1
    if fraction > 0:
        return str(whole)+'.'+str_fraction.rstrip()
    else:
        return str(whole)
def num2str(num_list, precision = 2):
    # This function returns a string of the form x_y_z_... where x,y,z are numbers with precision digits
    if precision == 0:
        l = 1
    else:
        l = precision+2
    string = '_'.join([fstr(i, precision) for i in num_list])
    return string
def num2expstr(num, precision = 2):
    # This function returns a string of the form xordery where x is the mantissa and y is the exponent of num with precision digits
    if precision == 0:
        l = 1
    else:
        l = precision+2
    # determine the exponent
    order = int(floor(log10(abs(num))))
    # determine the mantissa
    mantissa = num/10**order
    # Create the string
    if not order == 0:
        string = fstr(mantissa, precision)+'e'+fstr(order, 0)
    else:
        string = fstr(mantissa, precision)
    return string
def plusminus(num, std_num, precision = 2):
    # This function returns a string of the form (num±std_num)eorder	
    if precision == 0:
        l = 1
    else:
        l = precision+2
    # determine the exponent
    order = int(floor(log10(abs(num))))
    order_error = int(floor(log10(abs(std_num))))
    # determine precision so that at least one digit is shown of std_num
    if order_error < order:
        min_precision = order - order_error
        if precision < min_precision:
            precision = min_precision
    # determine the mantissa
    mantissa = num/10**order
    mantissa_std = std_num/10**order
    # Create the string
    if not order == 0:
        string = '('+fstr(mantissa, precision)+'±'+fstr(mantissa_std, precision)+')e'+fstr(order, 0)
    else:
        string = fstr(mantissa, precision)+'±'+fstr(mantissa_std, precision)
    return string

def rep_zip(*args):
    """
    A function that works like zip but also allows for inputting non iterables/object without length which it then repeats for every element of 
    the first iterable.
    Args:
    - args: positional arguments containing any combination of iterables and non-iterable objects
    
    Returns:
    - a generator object that yields tuples of values from the input iterables, or repeated non-iterable objects
    """
    # find an iterable or object with length
    if hasattr(args[0], '__iter__'):
            min_len = len(list(args[0]))
    else:
        raise ValueError('No iterable or object with length found in input')
    # Iterate over the first min_len elements of each iterable (or just the single non-iterable object)
    for i in range(min_len):
        yield *tuple(arg[i] if i==0 else arg for i, arg in enumerate(args)),

def rep_zip_iterables(*args):
    # returns iterators for every single element of the input iterables
    # first argument is used to determine the number of elements
    # subsequent ones get repeated
    if hasattr(args[0], '__iter__'):
            min_len = len(list(args[0]))
    else:
        raise ValueError('No iterable or object with length found in input')
    out_args = [args[0]]
    for i in range(1,len(args)):
        out_args.append([args[i]]*min_len)
    return *out_args,
        
### Custom Iterator based on nditer but adding slices to specified dimensions
# flatten As second and third axis using 
def nditer_slices(shape, slice_dimensions=[]):
    if isinstance(shape, tuple):
        shape = list(shape)
    elif isinstance(shape, int):
        shape = [shape]
    dims = shape.copy()
    for i in range(len(dims)):
        if i in slice_dimensions:
            dims[i] = 1
    product_input = []
    for i in range(len(dims)):
        if i in slice_dimensions:
            product_input.append([slice(None)])
        else:
            product_input.append(list(range(dims[i])))
    for indices in product(*product_input):
        yield indices

def mean_std_asym(arr, axis=-1):
    # Calculate the mean and two std's for each side of the mean (asymmetric std)
    # first check if axis is a tuple
    if isinstance(axis, (tuple, list, np.ndarray)):
        # create a shape_out list 
        shape = arr.shape
        shape_out = []
        for i in range(len(shape)):
            if not i in axis:
                shape_out.append(shape[i])
        mean = np.zeros(shape_out)
        std_min = np.zeros(shape_out)
        std_max = np.zeros(shape_out)
        for ind, ind_reduced in zip(nditer_slices(shape, slice_dimensions=axis), nditer_slices(shape_out)):
            mean[ind_reduced], std_min[ind_reduced], std_max[ind_reduced] = mean_std_asym(arr[ind], axis=-1) 
        return mean, std_min, std_max
    elif axis < 0:
        mean = np.mean(arr)
        smaller_than = arr[arr<mean]
        larger_than = arr[arr>mean]
        std_min = np.sqrt(np.sum((mean - smaller_than)**2)/(len(smaller_than) - 1))
        std_max = np.sqrt(np.sum((larger_than - mean)**2)/(len(larger_than) -1))
        return mean, std_min, std_max
    else:
        # iterate over all axis except the one specified in axis
        #define output arrays and their shape
        shape = arr.shape
        shape_out = shape[:axis] + shape[axis+1:]
        mean = np.zeros(shape_out)
        std_min = np.zeros(shape_out)
        std_max = np.zeros(shape_out)
        # iterate over all axis except the one specified in axis, get the index for every iteration in loop
        for ind in np.ndindex(shape_out):
            # modify the index to include slice along axis
            ind_post = ind[:axis] + (slice(None),) + ind[axis:]
            # calculate mean and asymmetric std for this slice
            mean[ind], std_min[ind], std_max[ind] = mean_std_asym(arr[ind_post], axis=-1)
        return mean, std_min, std_max
    
def std_min(arr):
    # Calculate the lower side std for asym std
    mean = np.mean(arr)
    smaller_than = arr[arr<mean]
    std_min = np.sqrt(np.sum((mean - smaller_than)**2)/(len(smaller_than) - 1))
    return std_min

def std_max(arr):
    # Calculate the upper side std for asym std
    mean = np.mean(arr)
    larger_than = arr[arr>mean]
    std_max = np.sqrt(np.sum((larger_than - mean)**2)/(len(larger_than) -1))
    return std_max

##### Batch functions ##########################################################
def batch_range(A, step=50, ind=0):
    # a function that returns batches of length step of a list A
    # batch_range is a generator that yields batches of length step of a list A -> [A[0:step], A[step:2*step], ...]
    # the last batch may be shorter than step
    n = len(A)
    while ind*step < n:
        yield A[ind*step:min([(ind+1)*step, n])]
        ind += 1
def batch_range_length(A, step=50):
    # a function that returns how many batches of length step a list A has
    n = len(A)
    how_many = n // step
    return how_many + (1 if n % step > 0 else 0)

# Batch processing
def batch_fun(fun, A, how_many_samples):
    # Calculates a batched version of fun
    fun_A = []
    for a in batch_range(A, how_many_samples):
        fun_A.append(fun(a))
    return fun_A

def batch_fun_dict(fun, dictionary, how_many_samples):
    dict = {}
    for key in dictionary.keys():
        dict[key] = {}
        for key2 in dictionary[key].keys():
            dict[key][key2] = batch_fun(fun, dictionary[key][key2], how_many_samples)
    return dict

def batch_mean_std_dict(dictionary, how_many_samples):
    dict_mean = batch_fun_dict(np.mean, dictionary, how_many_samples)
    dict_std = batch_fun_dict(np.std, dictionary, how_many_samples)
    return dict_mean, dict_std

def batch_mean_std_asym_dict(dictionary, how_many_samples):
    dict_mean = batch_fun_dict(np.mean, dictionary, how_many_samples)
    dict_std_min = batch_fun_dict(std_min, dictionary, how_many_samples)
    dict_std_max = batch_fun_dict(std_max, dictionary, how_many_samples)
    return dict_mean, dict_std_min, dict_std_max

# Make a list of all files that have been used in the cache folder
def add_cached_files_to_list(hash, cache_folder='Cache'):
    # add hash to list of hashes unless it is already in the list
    # if the list does not exist, create it
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)
    # if not file exists, create it
    hash_list_filename = os.path.join(cache_folder, 'hash_list.txt')
    if not os.path.exists(hash_list_filename):
        with open(hash_list_filename, 'w') as f:
            f.write(hash)
    else:
        with open(hash_list_filename, 'r') as f:
            hash_list = f.read().splitlines()
        if not hash in hash_list:
            with open(hash_list_filename, 'a') as f:
                f.write('\n' + hash)

# find all the files that don't contain the hash in the cache folder and move them to the Unused_Cache folder (create if needed)
def move_unused_cache_files(cache_folder='Cache', unused_folder='Unused_Cache'):
    has_list_filename = os.path.join(cache_folder, 'hash_list.txt')
    # load hash list
    with open(hash_list_filename, 'r') as f:
        hash_list = f.read().splitlines()
    # find all files in cache folder that are not in hash list
    files = os.listdir(cache_folder)
    # first select only the .npz files
    files = [f for f in files if f.endswith('.npz')]
    for hash in hash_list:
        files = [f for f in files if not hash in f]   # remove files from files list that are in hash list
    # move all files that are not in hash list to unused folder
    if not os.path.exists(unused_folder):
        os.makedirs(unused_folder)
    for f in tqdm(files, total=len(files)):
        shutil.move(os.path.join(cache_folder, f), os.path.join(unused_folder, f))
        
# find all files ending with npy in Cache folder, load them and rtesave the data as npz files
def npy2npz(dir='Cache'):
    files = os.listdir(dir)
    for file in files:
        if file.endswith('.npy'):
            print(file)
            data = np.load(os.path.join(dir, file), allow_pickle=True)
            var, calculated, how_many_missing = data
            new_file = file.replace('.npy', '.npz')
            np.savez(os.path.join(dir, new_file), var=var, calculated=calculated, how_many_missing=how_many_missing, allow_pickle=True)

# find the start and end indexes between two strings in a longer string
def find_between(s, first, last):
    try:
        start = s.index(first) + len(first)
        end = s.index(last, start)
        return start, end
    except ValueError:
        return None

def replace_between(s, first, last, new):
    try:
        start, end = find_between(s, first, last)
        return s[:start] + new + s[end:]
    except ValueError:
        return None
    
# a function that takes a list of line strings, and finds the starting and endpoint of a brace block with lines and line indexes, 
# and returns them
def find_brace_block(lines, start_str, brace_type=['{', '}']):
    if not start_str.endswith(brace_type[0]):
        start_str_before = start_str
        start_str += brace_type[0]
        print('added ' + brace_type[0] + ' to start_str ('+start_str_before +' -> ' + start_str +')')
    # find the start of the brace block, count opening and closing braces from it, until the count is zero again
    #find the first occurance of start_str
    start_line = -1
    end_line = -1
    for i in range(len(lines)):
        curr_index = 0
        if start_str in lines[i]:
            start_line = i
            # index 
            start_index = lines[i].index(start_str)
            brace_counter = 0
            # count braces in this line by counting letter by letter   
            curr_index = start_index  
            break
    found = False
    if start_line > -1:
        for i in range(start_line, len(lines)):
            if i > start_line:
                start_index = 0
            # find end of brace block
            open_braces_in_line = lines[i][start_index:].count(brace_type[0])
            close_braces_in_line = lines[i][start_index:].count(brace_type[1])
            new_brace_counter = brace_counter + open_braces_in_line - close_braces_in_line
            try_this_line = 0
            if new_brace_counter == 0: # ends in this line
                # element by element find the line index of the closing brace
                try_this_line = 1
            if close_braces_in_line > brace_counter:
                # brace could be closed here
                try_this_line = 1
            else:
                # brace is not closed here, so it must be closed in the next line
                pass    
                curr_counter = brace_counter
            if try_this_line:
                while curr_index < len(lines[i]):
                    if lines[i][curr_index] == brace_type[1]:
                        brace_counter -= 1
                        if brace_counter == 0:
                            # found the closing brace
                            end_line = i
                            end_index = curr_index
                            found = True
                            break
                    elif lines[i][curr_index] == brace_type[0]:
                        brace_counter += 1
                    curr_index += 1
            if not found:
                brace_counter = new_brace_counter
            else:
                break
        if start_line < 0:
            raise ValueError('start_str not found in lines')
        if end_line < 0:
            raise ValueError('brace block not closed in lines')
    else:
        raise ValueError('start_str not found in lines (' + start_str + ')')
    return start_line, start_index, end_line, end_index

def replace_between_in_brace_block(lines, start_str, first, last, new, brace_type=['{', '}']):
    start_line, start_index, end_line, end_index = find_brace_block(lines, start_str, brace_type)
    reduced_lines = lines[start_line:end_line+1]
    reduced_lines[0] = reduced_lines[0][start_index:]
    reduced_lines[-1] = reduced_lines[-1][:end_index+1]
    # find first occurance of first
    first_line = -1
    for i in range(len(reduced_lines)):
        if first in reduced_lines[i]:
            first_line = i
            break
    if first_line < 0:
        raise ValueError('first not found in reduced_lines')
    s = replace_between(reduced_lines[first_line], first, last, new)
    print(s)
    ind = start_line+first_line
    ind_start = 0
    if ind == start_line:
        ind_start = start_index
    lines[ind] = lines[ind][:start_index] + s 
    return lines

def replace_or_add_between_in_brace_block(lines, start_str, first, last, new, brace_type=['{', '}'], addition=''):
    start_line, start_index, end_line, end_index = find_brace_block(lines, start_str, brace_type)
    reduced_lines = lines[start_line:end_line+1]
    reduced_lines[0] = reduced_lines[0][start_index:]
    after_str = reduced_lines[-1][end_index+1:]
    reduced_lines[-1] = reduced_lines[-1][:end_index+1]
    # find first occurance of first
    first_line = -1
    for i in range(len(reduced_lines)):
        if first in reduced_lines[i]:
            first_line = i
            break
    if first_line < 0:
        if len(addition):
            # add in front of the closing brace
            lines[end_line] = lines[end_line][:end_index] + addition + lines[end_line][end_index:]
        else:
            raise ValueError('first not found in reduced_lines, and no addition given')
    else:
        s = replace_between(reduced_lines[first_line], first, last, new)
        ind = start_line+first_line
        ind_start = 0
        if ind == start_line:
            ind_start = start_index
        lines[ind] = lines[ind][:ind_start] + s + after_str
    return lines


# test
#lines = ['legend cell align={left},', 
#         'legend style={fill opacity=0.8,', 
#         'draw opacity=1,', 
#         'text opacity=1,' , 
#         'anchor=south west,',
#         'draw=lightgray204,',
#         'at={(0.5,1.05)}}, ']
#start_str = 'legend style='
#first = 'at={('
#last = ','
#new = '1.0'
#replace_between_in_brace_block(lines, start_str, first, last, new)


def clean_notebooks(dir=''):
    # Removes the output from all notebooks in the given directory and its subdirectories
    if dir == '':
        dir = os.getcwd()
    for root, dirs, files in os.walk(dir):
        # call nbstripout on each notebook
        for file in files:
            if file.endswith('.ipynb'):
                print('Cleaning ' + os.path.join(root, file))
                os.system('nbstripout ' + os.path.join(root, file))