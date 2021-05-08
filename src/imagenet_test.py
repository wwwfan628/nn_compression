from torchvision.datasets import ImageFolder
import torch
from torchvision import transforms


class ImageFolderWithPaths(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path




if __name__ == '__main__':

    num_workers = 8
    batch_size = 1
    transform_train = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_val = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    data_train = ImageFolderWithPaths(root='/srv/beegfs01/projects/imagenet/data/train/', transform=transform_train)
    data_test = ImageFolderWithPaths(root='/srv/beegfs01/projects/imagenet/data/val/', transform=transform_val)

    dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=False, pin_memory=True,
                                                   num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=False, pin_memory=True,
                                                  num_workers=num_workers)
    for i, (images, labels, paths) in enumerate(dataloader_train):
        print(paths)
        print('\n')
    for i, (images, labels, paths) in enumerate(dataloader_test):
        print(paths)
        print('\n')
    print("Finish!")