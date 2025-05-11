import cv2
import numpy as np

def order_pts(pts):
    """
    Return 4 points ordered as: top-left, top-right, bottom-right, bottom-left.
    """
    pts = pts.reshape(4, 2)
    s   = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")

def expand_quad(quad, margin_pct=0.03):
    """
    Uniformly scale quad outward about its centroid.
    """
    centroid = quad.mean(axis=0, keepdims=True)
    scale    = 1.0 + margin_pct
    return (quad - centroid) * scale + centroid

def extract_card(img_path, out_path="card_crop.png",
                 debug=False, margin_pct=0.03):
    """
    Detects a trading card in a snapshot, warps it square, and saves `out_path`.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(img_path)
    orig = img.copy()
    H, W = img.shape[:2]

    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges  = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    card_ratio = 2.5 / 3.5
    candidate  = None

    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(c)
        if area < 0.01 * H * W:
            continue

        rect   = cv2.minAreaRect(c)
        (w, h) = rect[1]
        if w == 0 or h == 0:
            continue

        ratio = max(w, h) / min(w, h)
        if 1.15 <= ratio <= 1.7:
            candidate = cv2.boxPoints(rect)  
            break

    if candidate is None:
        raise RuntimeError("Couldn't locate a card")

    src_pts = order_pts(expand_quad(candidate, margin_pct))
    dst_w   = 300                              
    dst_h   = int(dst_w / card_ratio)          
    dst_pts = np.array([[0, 0],
                        [dst_w - 1, 0],
                        [dst_w - 1, dst_h - 1],
                        [0, dst_h - 1]], dtype="float32")

    M       = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warp    = cv2.warpPerspective(orig, M, (dst_w, dst_h))

    cv2.imwrite(out_path, warp)

    if debug:
        debug_img = orig.copy()
        cv2.drawContours(debug_img, [candidate.astype(int)], -1,
                         (0, 255, 0), 3)
        cv2.imshow("Detected rectangle", debug_img)
        cv2.imwrite('debug_img.jpg', debug_img)
        cv2.imshow("Warped card", warp)
        cv2.waitKey(0); cv2.destroyAllWindows()

    print(f"Saved: {out_path}")
    return warp

if __name__ == "__main__":
    in_path  = "IMG_9245.jpg"
    out_path = "card_crop.jpg"
    extract_card(in_path, out_path, debug=True)
