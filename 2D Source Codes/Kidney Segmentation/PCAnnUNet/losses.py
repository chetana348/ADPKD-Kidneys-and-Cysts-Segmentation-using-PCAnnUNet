from keras_unet import TF
if TF:
    from tensorflow.keras import backend as K
else:
    from keras import backend as K


def Hausdorff_distance(y_true, y_pred, smooth=100):
    
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    Haus = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - Haus) * smooth
