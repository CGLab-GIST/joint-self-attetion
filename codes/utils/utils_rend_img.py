# Some functions of this script came from the repository of MC-Denoising-via-Auxiliary-Feature-Guided-Self-Attention (https://github.com/Aatr0x13/MC-Denoising-via-Auxiliary-Feature-Guided-Self-Attention).

import numpy as np

eps = 1e-8

def preprocess_depth(depth):
    depth = np.clip(depth, 0.0, np.max(depth))
    max_feature = np.max(depth)
    if max_feature != 0:
        depth /= max_feature
    return depth


def preprocess_normal(normal):
    normal = np.nan_to_num(normal)
    normal = (normal + 1.0) * 0.5
    normal = np.maximum(np.minimum(normal, 1.0), 0.0)
    return normal


def preprocess_specular(specular):
    # assert np.sum(specular < 0) == 0, "Negative value in specular component!"
    return np.log(specular + 1)


def postprocess_specular(specular):
    return np.exp(specular) - 1


def postprocess_diffuse(diffuse, albedo):
    return diffuse * (albedo + eps)

def tone_mapping(matrix, gamma=2.2):
    return np.clip(matrix ** (1.0 / gamma), 0, 1)

def tensor2img(image_numpy, post_spec=False, post_diff=False, albedo=None):
    if post_diff:
        assert albedo is not None, "must provide albedo when post_diff is True"
    image_type = np.uint8

    # multiple images
    if image_numpy.ndim == 4:
        temp = []
        for i in range(len(image_numpy)):
            if post_diff:
                temp.append(tensor2img(image_numpy[i], post_spec=False, post_diff=post_diff, albedo=albedo[i]))
            else:
                temp.append(tensor2img(image_numpy[i], post_spec=post_spec))
        return np.array(temp)
    image_numpy = np.transpose(image_numpy, (1, 2, 0))

    # postprocessing
    if post_spec:
        image_numpy = postprocess_specular(image_numpy)
    elif post_diff:
        albedo = np.transpose(albedo, (1, 2, 0))
        image_numpy = postprocess_diffuse(image_numpy, albedo)
    image_numpy = tone_mapping(image_numpy) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255).astype(image_type)

    return image_numpy