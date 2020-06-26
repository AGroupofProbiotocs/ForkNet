import numpy as np
from imgaug import augmenters as iaa

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.
chance = 1./6
seq = iaa.Sequential([
        iaa.Sometimes(chance,iaa.Fliplr(1)), # horizontally flip 50% of all images
        iaa.Sometimes(chance,iaa.Flipud(1)), # vertically flip 50% of all images
        iaa.Sometimes(chance*4, iaa.OneOf([iaa.Affine(rotate=0),
                                         iaa.Affine(rotate=90),
                                         iaa.Affine(rotate=180),
                                         iaa.Affine(rotate=270) # rotate by (90,180,270) degrees
                                         ]))],
                    random_order=True # do all of the above in random order
                    )

def patch_batch_generator(Y, label, batch_size=64, patch_width=64, patch_height=64, random_shuffle=True, augment=True):
    '''
    A generator that yields a batch of (data, label).

    Input:
        data_list : a number of image indexes
        random_shuffle : whether shuffle rows in the data_llist
        batch_size : batch size

    Output:
        (X_batch, Y_batch)
    '''
    
    seq_fixed = seq.to_deterministic()
    
    N = len(Y)
    # note that X is the output demosaiced image, while Y is the input image.
    # remove the last channel of label, which was added for saving.


    if random_shuffle:
        index = np.random.permutation(N)
    else:
        index = np.arange(N)
    
    #count the batch index
    batch_index = 0
    while True:
        current_index = (batch_index * batch_size) % N
        #if reach the last batch，change the batch size to prevent not enough samples
        if N >= (current_index + batch_size):
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0

        Y_batch = np.zeros((current_batch_size, patch_width, patch_height, Y.shape[-1]))

#        X_batch = np.zeros((current_batch_size, patch_width, patch_height, 4))
        label_batch = np.zeros((current_batch_size, patch_width, patch_height, label.shape[-1]))
        

        for i in range(current_index, current_index + current_batch_size):
            #fill the current images into the batch
            Y_batch[i - current_index] = Y[index[i]]
#            X_batch[i - current_index] = X[index[i]]
            label_batch[i - current_index] = label[index[i]]
        
        if augment:
            Y_batch = seq_fixed.augment_images(Y_batch)
#            X_batch = seq_fixed.augment_images(X_batch)
            label_batch = seq_fixed.augment_images(label_batch)
                
#        if normalize:
#            X_left_batch = X_left_batch.astype(np.float64)
#            X_left_batch = preprocess_input(X_left_batch)
        

        yield (Y_batch, label_batch)


