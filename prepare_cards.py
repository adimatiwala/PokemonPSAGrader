from PIL import Image
import os 
import numpy as np


# Following Code took references from 
def crop_images(dir, grade):
    """
    The images have a grade at the top; automatically remove the grade.

    Parameters:
    -------
    dir: folder directory of images
    grade: the PSA Grade

    Returns:
    --------
    None

    Iterates through the directory and crops the score out of the image
        1) Removes images whose width are not 300 ± 5px wide as the scraped images all have a height of 500px
        2) If the images is valid it crops the top 1/5 of the image
        3) Saves the new image

    This is a simple method at filtering images that are not in the proper format. 

    """
    counter = 0   
    n = len(os.listdir(dir))
    for fname in os.listdir(dir):

        try:
            # Counter
            counter += 1
            if counter % 100 == 0:
                print("{}/{}".format(counter, n))

            # Load image as numpy array.
            p = os.path.join(dir, fname)
            img = np.array(Image.open(p))

            height, width = img.shape[:2]

            # Removing images not in the proper format
            if not (295 <= width  <= 305 and
                    495 <= height <= 505):
                continue

            new_top = int(height/5)           
            cropped = img[new_top:, :, :]

            # Downloading the images
            os.makedirs("samples", exist_ok=True)
            folder_path = f"samples/psa{grade}"
            os.makedirs(folder_path, exist_ok=True)
            of_p = os.path.join(folder_path, fname)
            PIL_img = Image.fromarray(cropped)
            PIL_img.save(of_p)

        except Exception as e:

            print(e)

if __name__ == "__main__":
    paths = ["psa1_images","psa2_images","psa3_images","psa4_images","psa5_images","psa6_images","psa7_images","psa8_images","psa9_images","psa10_images"]
    grade = 1
    for path in paths:
        crop_images(f"raw_data/{path}", grade)
        grade += 1