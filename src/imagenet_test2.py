from pathlib import Path
from PIL import Image
import PIL


if __name__ == '__main__':

    print("Test Training Dataset")
    base_path = Path('/srv/beegfs01/projects/imagenet/data/train/')
    for class_dir in list(base_path.iterdir()):
        class_dir_path = base_path / class_dir
        for image_file_name in list(class_dir_path.iterdir()):
            image_path = str(class_dir_path / image_file_name)
            try:
                image = Image.open(image_path)
                image.verify()
            except (OSError, PIL.UnidentifiedImageError) as e:
                print('Corrupted Image: ', image_path)

    print("Test Validation Dataset")
    base_path = Path('/srv/beegfs01/projects/imagenet/data/val/')
    for class_dir in list(base_path.iterdir()):
        class_dir_path = base_path / class_dir
        for image_file_name in list(class_dir_path.iterdir()):
            image_path = str(class_dir_path / image_file_name)
            try:
                image = Image.open(image_path)
                image.verify()
            except (OSError, PIL.UnidentifiedImageError) as e:
                print('Corrupted Image: ', image_path)
    print("Finish!")