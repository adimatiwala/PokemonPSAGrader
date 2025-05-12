# Pokémon Card PSA Grader

Contributors: Aditya Matiwala, Adrian Lara, Kashvi Panjolia

To upload your own Pokémon card for grading by our model, clone the repository into your environment. Then, download the following packages using the pip installer:

* PyTorch
* Selenium
* OpenCV
* Numpy
* Tqdm
* Scikit-Image
* Albumentations

Next, upload your image to **INSERT DIRECTORY HERE** for grading. Run the image through the prepare_cards.py file, then the extract_card.py file so the program will know where your card is located in the image. Next, run the preprocess_cards.py file to clean up the image and ensure it will be readable by this model.


Finally, run the image through train_model.py to run the model on your image and obtain a PSA score for your own Pokémon card.
