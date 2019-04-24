

import numpy as np
import argparse
import os
import torchvision
import torchvision.transforms as transforms
import torch
import scipy
import classify_svhn
from classify_svhn import Classifier
from scipy import linalg

SVHN_PATH = "svhn"
PROCESS_BATCH_SIZE = 32


def get_sample_loader(path, batch_size):
    """
    Loads data from `[path]/samples`

    - Ensure that path contains only one directory
      (This is due ot how the ImageFolder dataset loader
       works)
    - Ensure that ALL of your images are 32 x 32.
      The transform in this function will rescale it to
      32 x 32 if this is not the case.

    Returns an iterator over the tensors of the images
    of dimension (batch_size, 3, 32, 32)
    """
    data = torchvision.datasets.ImageFolder(
        path,
        transform=transforms.Compose([
            transforms.Resize((32, 32), interpolation=2),
            classify_svhn.image_transform
        ])
    )
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        num_workers=2,
    )
    return data_loader


def get_test_loader(batch_size):
    """
    Downloads (if it doesn't already exist) SVHN test into
    [pwd]/svhn.

    Returns an iterator over the tensors of the images
    of dimension (batch_size, 3, 32, 32)
    """
    testset = torchvision.datasets.SVHN(
        SVHN_PATH, split='test',
        download=True,
        transform=classify_svhn.image_transform
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
    )
    
    return testloader


def extract_features(classifier, data_loader):
    """
    Iterator of features for each image.
    """
   
    with torch.no_grad():
        for x, _ in data_loader:
            h = classifier.extract_features(x).numpy()
            for i in range(h.shape[0]):
                yield h[i]


def calculate_fid_score(sample_feature_iterator, testset_feature_iterator):
# based on https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py
    
    f = np.zeros((512, 1000)) 
    r = np.zeros((512, 1000)) 
    # normalize extracted features by dividing by their max value to get 0 to 1
    for i, x in enumerate(sample_feature_iterator):
            
        f[:,i] = next(sample_feature_iterator)
        min_max = np.max(np.absolute(f[:,i]))
        f[:,i] = f[:,i]/min_max
 
        r[:,i] = next(testset_feature_iterator)
        min_max = np.max(np.absolute(r[:,i]))
        r[:,i] = r[:,i]/min_max
     
    fake = f[:,:i]
    real = r[:,:i]
    
    print(i)
    
    mu_q = np.mean(fake, axis=0) 
    mu_p = np.mean(real, axis=0) 
        
    sigma_q = np.cov(fake, rowvar=False)
    sigma_p = np.cov(real, rowvar=False)
    
    covmean, _ = linalg.sqrtm(sigma_p.dot(sigma_q), disp=False)
    
    if not np.isfinite(covmean).all(): # avoid nan
        offset = np.eye(sigma_p.shape[0]) * 1e-5
        covmean = linalg.sqrtm((sigma_p + offset).dot(sigma_q + offset))

    covmean = covmean.real
    
    meandiff = np.dot(mu_p-mu_q,mu_p-mu_q)
    
    trace_sigma_p = np.trace(sigma_p)
    trace_sigma_q = np.trace(sigma_q)
    trace_cov = np.trace(covmean)
        
    return meandiff + trace_sigma_p + trace_sigma_q - 2*trace_cov
  
  
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Score a directory of images with the FID score.')
    parser.add_argument('--model', type=str, default="svhn_classifier.pt",
                        help='Path to feature extraction model.')
    parser.add_argument('directory', type=str,
                        help='Path to image directory')
    args = parser.parse_args()

    quit = False
    if not os.path.isfile(args.model):
        print("Model file " + args.model + " does not exist.")
        quit = True
    if not os.path.isdir(args.directory):
        print("Directory " + args.directory + " does not exist.")
        quit = True
    if quit:
        exit()
    print("Test")
    classifier = torch.load(args.model, map_location='cpu')
    classifier.eval()

    sample_loader = get_sample_loader(args.directory,
                                      PROCESS_BATCH_SIZE)
    sample_f = extract_features(classifier, sample_loader)

    test_loader = get_test_loader(PROCESS_BATCH_SIZE)
    test_f = extract_features(classifier, test_loader)

    fid_score = calculate_fid_score(sample_f, test_f)
    print("FID score:", fid_score)
   
    #VAE FID score: 28.954747153107675 GAN FID score: 29.90769304481418