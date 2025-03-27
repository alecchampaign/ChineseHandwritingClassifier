from fastai.vision.all import *
import kaggle

def download_kaggle_dataset():
    try:
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files('pascalbliem/handwritten-chinese-character-hanzi-datasets')
    except Exception as e:
        print(f"Error downloading dataset: {e}")

path = download_kaggle_dataset()
dls = ImageDataLoaders.from_folder(path, get_image_files(path), valid_pct=0.2, seed=42, item_tfms=Resize(224))

learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(10)