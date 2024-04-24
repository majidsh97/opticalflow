import numpy as np

def computeBilinerWeights(q):
    ## TODO 2.1
    ## - Compute bilinear weights for point q
    ## - Entry 0 weight for pixel (x, y)
    ## - Entry 1 weight for pixel (x + 1, y)
    ## - Entry 2 weight for pixel (x, y + 1)
    ## - Entry 3 weight for pixel (x + 1, y + 1)
    weights = [1, 0, 0, 0]
    x , y = q

    x_int = int(x)
    y_int = int(y)

    a = x - x_int
    b = y - y_int

    weight_tl = (1 - a) * (1 - b)
    weight_tr = a * (1 - b)
    weight_bl = (1 - a) * b
    weight_br = a * b

    weights = [weight_tl, weight_tr, weight_bl, weight_br]
    #     I(q) =(1 - a)(1 - b) . I(x; y)
    # a(1 - b) . I(x + 1; y)
    # (1 - a)b . I(x; y + 1)
    # ab . I(x + 1; y + 1)
    return weights

def computeGaussianWeights(winsize, sigma):

    ## TODO 2.2
    ## - Fill matrix with gaussian weights
    ## - Note, the center is ((winSize.width - 1) / 2,winSize.height - 1) / 2)
    width, height = winsize
    weights = []
    cx = (width-1) / 2
    cy = (height - 1) / 2
    weights = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            normalized_x = (x - cx) / width
            normalized_y = (y - cy) / height
            gaussian_weight = np.exp(-(normalized_x ** 2 + normalized_y ** 2) / (2 * (sigma ** 2)))
            weights[y, x] = gaussian_weight

    return np.array(weights)


def invertMatrix2x2(A):

    ## TODO 2.3
    ## - Compute the inverse of the 2 x 2 Matrix A
    return np.linalg.inv(A)

