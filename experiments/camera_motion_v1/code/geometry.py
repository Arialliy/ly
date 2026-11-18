"""OpenCV integer pixel-center homographies; bilinear float32 only."""
import cv2
import numpy as np

MEAN = 85.0799
RF_RADIUS = 9  # 7x7 input convolution + two (3x3, dilation-2 3x3) blocks.

def warp(a, H, fill=0.):
    h,w=a.shape
    return cv2.warpPerspective(np.asarray(a,np.float32), np.asarray(H,np.float64),
        (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=float(fill))

def erode(a):
    return cv2.erode(a.astype(np.uint8), np.ones((2*RF_RADIUS+1,)*2,np.uint8),
                     borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)

def prepare(raw, transforms):
    """Return C1 raw-domain input, C2 anchor input, common output validity.

    Every valid output excludes padding from all 40 temporal inputs and the
    network spatial receptive field. Same mask is used in both conditions.
    """
    ones=np.ones(raw.shape[1:],np.float32)
    aligned=[]; roundtrip=[]; va=[]; vr=[]
    for img,H in zip(raw,transforms):
        inv=np.linalg.inv(H)
        a=warp(img,H,MEAN); support=warp(ones,H)>=1.-1e-6
        r=warp(a,inv,MEAN); support_r=warp(support.astype(np.float32),inv)>=1.-1e-6
        aligned.append(a);roundtrip.append(r);va.append(support);vr.append(support_r)
    anchor_valid=erode(np.logical_and.reduce(va))
    original_valid=erode(np.logical_and.reduce(vr))
    common=[]
    for H in transforms:
        restored=warp(anchor_valid.astype(np.float32),np.linalg.inv(H))>=1.-1e-6
        common.append(restored & original_valid)
    return np.stack(roundtrip),np.stack(aligned),np.stack(common)

def restore(probabilities, transforms):
    return np.stack([warp(p,np.linalg.inv(H),0.) for p,H in zip(probabilities,transforms)])

def apply_fallback(candidate, original, valid):
    # Geometry-only choice. Never choose the larger of original/candidate scores.
    return np.where(valid,candidate,original).astype(np.float32)
