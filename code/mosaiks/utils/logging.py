import time
import warnings
import os
#import grp

from mosaiks import config as c

def robust_chmod(path, permission=0o774):
    """
    Helper function to avoid chmod erroring out if you try to chmod a path you
    did not create.
    """
    return True
#     try:
#         os.chmod(path, permission)
#     except PermissionError:
#         log_text("Cannot change permission for {}.".format(path))

#     try:
#         os.chown(path, -1, grp.getgrnam("maps100").gr_gid)
    
#     except PermissionError:
#         log_text("Cannot change group for {}.".format(path))

def log_text(text, mode="a", savepath = "", filename="", print_text= c.verbose): 
    """
    Function that writes text to a log file. Takes text and mode as inputs. Default mode is "a" for append. Can also accept "w" which will clear the log.
    """
    print(text)
    return True
#     if savepath in [None, "", False]:
#         savepath = getattr(c,'log_savedir', getattr(c,'main_log_dir'))
        
#     if print_text == "warn":
#         warnings.warn(text)
#     elif print_text:
#         print(text)
        
#     if filename:
#         filepath = savepath +'/'+ time.strftime("%Y%m%d_%H%M")  + "_" + filename + ".txt"
#         run_chmod = (not os.path.exists(filepath))

#         f = open(filepath, mode)

#     else:
#         filepath = savepath +'/' + time.strftime("%Y%m%d_%H%M")  + "log.txt"
#         run_chmod = (not os.path.exists(filepath))
#         f = open(filepath, mode)
#     f.write(("\n" + text))
    
#     if run_chmod:
#         robust_chmod(filepath)
#     f.close()

