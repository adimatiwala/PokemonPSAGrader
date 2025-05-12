import cv2
import numpy as np
import os
from tqdm import tqdm
from skimage import filters, exposure, restoration
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import matplotlib.pyplot as plt
import glob
import seaborn as sns

def enhance_image(img):
    """Enhance image quality with various preprocessing steps."""
    # Convert to LAB color space for better color processing
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Merge channels and convert back to BGR
    enhanced_lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Apply bilateral filter for noise reduction while preserving edges
    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    return enhanced

def normalize_colors(img):
    """Normalize colors to handle different lighting conditions."""
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Normalize L channel
    l = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX)
    
    # Merge channels and convert back to BGR
    normalized_lab = cv2.merge([l, a, b])
    normalized = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)
    
    return normalized

def detect_card_edges(img):
    """Enhanced card edge detection with multiple methods."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)
    
    # Apply morphological operations to clean up the image
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours, thresh

# Helper function to order points for perspective transform
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# Detect the largest contour (assumed to be the card), and align it
def detect_and_align_card(img, size=(224, 224)):
    # Enhance image quality first
    enhanced_img = enhance_image(img)
    normalized_img = normalize_colors(enhanced_img)
    
    # Try multiple edge detection methods
    contours, thresh = detect_card_edges(normalized_img)
    
    if contours:
        # Find the largest contour
        card_contour = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(card_contour, True)
        approx = cv2.approxPolyDP(card_contour, 0.02 * peri, True)
        
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            rect = order_points(pts)
            
            dst = np.array([
                [0, 0],
                [size[0] - 1, 0],
                [size[0] - 1, size[1] - 1],
                [0, size[1] - 1]
            ], dtype="float32")
            
            M = cv2.getPerspectiveTransform(rect, dst)
            aligned = cv2.warpPerspective(normalized_img, M, size)
            return aligned, thresh
    return None, None

def analyze_centering(img):
    """Analyze card centering by detecting borders and measuring symmetry."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to detect borders
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour (should be the card)
        card_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(card_contour)
        
        # Calculate centering metrics
        center_x = x + w/2
        center_y = y + h/2
        img_center_x = img.shape[1]/2
        img_center_y = img.shape[0]/2
        
        # Calculate offset from center
        x_offset = abs(center_x - img_center_x) / img_center_x
        y_offset = abs(center_y - img_center_y) / img_center_y
        
        return 1 - (x_offset + y_offset)/2
    return 0

def analyze_corners(img):
    """Analyze corner sharpness and wear."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect corners using Harris corner detector
    corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
    corners = np.int0(corners)
    
    # Calculate corner sharpness
    corner_sharpness = []
    for corner in corners:
        x, y = corner.ravel()
        if 0 < x < gray.shape[1]-1 and 0 < y < gray.shape[0]-1:
            # Calculate gradient magnitude at corner
            dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(dx[y,x]**2 + dy[y,x]**2)
            corner_sharpness.append(magnitude)
    
    return np.mean(corner_sharpness) if corner_sharpness else 0

def analyze_edges(img):
    """Analyze edge whitening and wear."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Calculate edge density and uniformity
    edge_density = np.mean(edges)
    
    # Analyze edge continuity
    edge_continuity = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    return {
        'edge_density': edge_density,
        'edge_continuity': edge_continuity
    }

def analyze_surface(img):
    """Analyze surface clarity and defects."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate surface texture
    texture = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Detect scratches and surface defects
    edges = cv2.Canny(gray, 50, 150)
    defect_density = np.mean(edges)
    
    return {
        'texture': texture,
        'defect_density': defect_density
    }

# Extract features from aligned cards
def extract_features(img):
    features = {}
    
    # Basic features
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features['blurriness'] = cv2.Laplacian(gray, cv2.CV_64F).var()
    features['brightness'] = np.mean(gray)
    features['sharpness'] = np.max(filters.sobel(gray))
    
    # Advanced features
    features['centering'] = analyze_centering(img)
    features['corner_sharpness'] = analyze_corners(img)
    
    edge_features = analyze_edges(img)
    features.update(edge_features)
    
    surface_features = analyze_surface(img)
    features.update(surface_features)
    
    return features

# Preprocess all images in the dataset
def preprocess_dataset(input_dir='samples', output_dir='aligned_samples', size=(224, 224)):
    os.makedirs(output_dir, exist_ok=True)
    features_dict = {}
    
    for psa_grade in range(1, 11):
        src_dir = os.path.join(input_dir, f'psa{psa_grade}')
        dst_dir = os.path.join(output_dir, f'psa{psa_grade}')
        os.makedirs(dst_dir, exist_ok=True)
        
        img_names = os.listdir(src_dir)
        
        for img_name in tqdm(img_names, desc=f'Processing PSA {psa_grade}'):
            img_path = os.path.join(src_dir, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Failed to load {img_path}")
                continue
            
            aligned_img, thresh = detect_and_align_card(img, size)
            
            if aligned_img is not None:
                # Save aligned image
                cv2.imwrite(os.path.join(dst_dir, img_name), aligned_img)
                
                # Save threshold image for debugging
                if thresh is not None:
                    thresh_path = os.path.join(dst_dir, f"thresh_{img_name}")
                    cv2.imwrite(thresh_path, thresh)
                
                # Extract and save features
                features = extract_features(aligned_img)
                features_dict[img_name] = features
            else:
                print(f"Alignment failed: {img_path}")
    
    return features_dict

def visualize_data(features_dict):
  img_to_psa = {}
  for psa_grade in range(1, 11):
      # features_dict doesn't store the psa grade with the image, so tracing back
      # through the paths to obtain the psa grade via the folder name
      image_paths = glob.glob(f'aligned_samples/psa{psa_grade}/*.jpg')
      for path in image_paths:
          filename = os.path.basename(path)
          img_to_psa[filename] = psa_grade


  df = pd.DataFrame.from_dict(features_dict, orient='index')
  df['psa_grade'] = df.index.map(img_to_psa)

  df = df.dropna(subset=['psa_grade'])

  # Convert to int
  df['psa_grade'] = df['psa_grade'].astype(int)

  # plot corner sharpness vs PSA grade
  plt.figure(figsize=(10, 6))
  sns.boxplot(x='psa_grade', y='corner_sharpness', data=df)
  plt.title('Corner Sharpness Distribution by PSA Grade')
  plt.xlabel('PSA Grade')
  plt.ylabel("Corner Sharpness")
  plt.show()

if __name__ == '__main__':
    features = preprocess_dataset()
    print("Feature extraction completed. Sample features:")
    for img_name, feat in list(features.items())[:3]:
        print(f"\nImage: {img_name}")
        for key, value in feat.items():
            print(f"{key}: {value:.4f}")
    visualize_data(features)
