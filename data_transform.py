import torchvision.transforms as transforms
from torchvision.transforms import functional as F

def prepare_transforms(args):
    train_transforms = []
    validation_transforms = []
    
    # Resize image
    validation_transforms.append(transforms.Resize(args.val_image_size + 32))
    validation_transforms.append(transforms.CenterCrop((args.val_image_size, args.val_image_size)))
            
    # RandomSizedCrop image
    train_transforms.append(transforms.RandomResizedCrop((args.train_random_sized_crop, args.train_random_sized_crop)))
        
    # Random flip
    train_transforms.append(transforms.RandomHorizontalFlip())
        
    # To tensor
    train_transforms.append(transforms.ToTensor())
    validation_transforms.append(transforms.ToTensor())

    # Normalize
    train_transforms.append(transforms.Normalize(args.mean, args.std,))
    validation_transforms.append(transforms.Normalize(args.mean, args.std,))

    data_transforms = {
        "train": transforms.Compose(train_transforms),
        "val": transforms.Compose(validation_transforms),
    }   
    return data_transforms

if __name__ == '__main__':
    from PIL import Image
    import numpy as np
    trans = [transforms.RandomCrop(224), transforms.Resize(224)]
    t = transforms.Compose(trans)
    a = Image.fromarray(np.random.rand(224, 224, 3).astype(np.uint8))
    a = t(a)