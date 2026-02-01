import time

def timed(function, *args):
    """
    Measure execution time of a grayscale function.
    function: The decolorization function to be measured.
    *args: original color image.
    returns: tuple (output, duration): The function result and time in seconds.
    Example: img_gray, t = timed(luminance_gray, img)
    """
    t0 = time.perf_counter()
    output = function(*args)
    t1 = time.perf_counter()
    return output, (t1 - t0)
