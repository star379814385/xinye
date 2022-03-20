

def num_clip(x, x_min, x_max):
    if x < x_min:
        return x_min
    elif x > x_max:
        return x_max
    return x
