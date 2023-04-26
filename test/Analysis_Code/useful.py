import os
import shutil
from numpy import floor, log10, abs

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

