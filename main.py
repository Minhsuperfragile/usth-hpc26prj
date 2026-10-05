from numba import cuda
from matplotlib.image import imread, imsave
import numpy as np
import math
import sys

BLOCK_SIZE = 16
R = 21
k = 7
h = 10

SM_SIZE = BLOCK_SIZE + (R-1) + (k-1)

def load_image(fname):
    img = imread(fname)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = img[:, :, :3]
    img = img.astype(np.float32)
    if img.max() <= 1.0:
        img *= 255.0
    padded = np.pad(img, pad_width=((13, 13), (13, 13), (0, 0)), mode="reflect")
    return img, padded

@cuda.jit
def nlm_denoise(padded_img, out):
    x,y = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # map pixel from global image to the shared image
    tid = ty * BLOCK_SIZE + tx

    shared_mem = cuda.shared.array(shape=(SM_SIZE,SM_SIZE,3), dtype=np.float32)
    for i in range(tid, SM_SIZE**2, BLOCK_SIZE**2):
        sy = i // SM_SIZE # shared img position
        sx = i % SM_SIZE
        gy = cuda.blockIdx.y * BLOCK_SIZE + sy # global image position
        gx = cuda.blockIdx.x * BLOCK_SIZE + sx
        if gy < padded_img.shape[0] and gx < padded_img.shape[1]:
            for c in range(3):  
                shared_mem[sy, sx, c] = padded_img[gy, gx, c] 
    cuda.syncthreads()

    if x < out.shape[1] and y < out.shape[0]:
        cx, cy = tx + 13, ty + 13 # current real pixel
        w = 0.
        r = g = b = 0.
        for ry in range(-10, 11): # loop the  S
            for rx in range(-10,11):
                jy, jx = cy + ry, cx + rx # each j in S
                d2 = 0.
                for py in range(-3,4): # loop the P
                    for px in range(-3,4): 
                        ky, kx = jy + py, jx + px # each k in P
                        for c in range(3): 
                            diff = shared_mem[cy + py, cx + px, c] - shared_mem[ky,kx,c]
                            d2 += diff ** 2 

                w_ = math.exp(-d2/(h**2))
                w += w_
                r += w_ * shared_mem[jy,jx,0]
                g += w_ * shared_mem[jy,jx,1]
                b += w_ * shared_mem[jy,jx,2]

        out[y,x,0] = r / w
        out[y,x,1] = g / w
        out[y,x,2] = b / w


def main():
    if len(sys.argv) < 2:
        sys.exit(1)

    img, padded = load_image(sys.argv[1])
    H, W = img.shape[:2]

    d_padded = cuda.to_device(np.ascontiguousarray(padded))
    d_out = cuda.device_array((H, W, 3), dtype=np.float32)

    threads = (BLOCK_SIZE, BLOCK_SIZE)
    blocks = ((W + BLOCK_SIZE - 1) // BLOCK_SIZE,
              (H + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    nlm_denoise[blocks, threads](d_padded, d_out)
    cuda.synchronize()

    out = d_out.copy_to_host()
    out = out.astype(np.uint8)
    imsave("output.jpg", out)


if __name__ == "__main__":
    main()