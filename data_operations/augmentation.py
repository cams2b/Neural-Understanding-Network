import matplotlib.pyplot as plt
from imgaug import augmenters as iaa

def augGeo():
    return iaa.SomeOf((0, 4),
                    [
                    iaa.CropAndPad(percent=(-0.10, 0.10)),
                    iaa.Fliplr(0.5),
                    iaa.Flipud(0.5),
                    iaa.Affine(scale={"x": (0.8, 1.5), "y": (0.8, 1.5)}, mode="constant", cval=-1024, order=[0,1]),
                    iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, mode="constant", cval=-1024, order=[0,1]),
                    iaa.Affine(rotate=(-90, 90), mode="constant", cval=-1024, order = [0,1]),
                    iaa.Affine(shear=(-10, 10)),
                    ], random_order=True)



def augInten():
    return iaa.SomeOf((0, 1),
                [
                iaa.Add((-0.05, 0.05)),
                iaa.Multiply((0.8, 1.2), per_channel=0.5),
                iaa.SaltAndPepper(0.03),
                iaa.OneOf([
                    iaa.LinearContrast((0.9, 1.1), per_channel=0.5),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.85, 1.25)),
                ]),
                iaa.OneOf([
                    iaa.Dropout((0.0, 0.005)),
                    iaa.SaltAndPepper((0.0, 0.005)),
                    iaa.GaussianBlur(sigma=(0.0, 2.0)),
                    iaa.AverageBlur(k=(2, 7)),
                    iaa.MedianBlur(k=(3, 5)),
                ]),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.25 * 1), per_channel=0.5),
            ], random_order=True)


def apply_aug(image_batch):
    geo_seq = augGeo()
    inten_seq = augInten()

    geo_seq = geo_seq.to_deterministic()
    inten_seq = inten_seq.to_deterministic()

    image_batch = geo_seq(images=image_batch)
    image_batch = inten_seq(images=image_batch)
   
    return image_batch




