from sklearn.datasets import fetch_openml
import numpy as np
from PIL import Image

def check_orientation():
    print("Fetching one batch of EMNIST...")
    X, y = fetch_openml('EMNIST_Balanced', version=1, return_X_y=True, as_frame=False, parser='auto')
    
    img_flat = X[0]
    label = y[0]
    
    img = img_flat.reshape(28, 28)
    
    print(f"Label: {label}")
    
    im = Image.fromarray(img.astype('uint8'))
    im.save("emnist_sample_original.png")
    print("Saved emnist_sample_original.png")
    
    img_T = img.T
    im_T = Image.fromarray(img_T.astype('uint8'))
    im_T.save("emnist_sample_transposed.png")
    print("Saved emnist_sample_transposed.png")

if __name__ == "__main__":
    check_orientation()
