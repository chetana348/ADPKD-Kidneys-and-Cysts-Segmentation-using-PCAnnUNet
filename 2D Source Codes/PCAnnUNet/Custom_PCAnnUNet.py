import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.base import RegressorMixin
from sklearn.base import ClassifierMixin
class SupervisedPCA(object):
    def __init__(self, fit_intercept=True, model=None,threshold=0,n_components=-1):
        self.fit_intercept = fit_intercept
        self._model=model
        
        self._pca=None
        self._leavouts=None
        self._threshold=threshold
        self._n_components=n_components
        
    def fit(self,X,y):
        """
        Fit the supervised PCA model
        .
        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples,n_features]
            Training data
        y : numpy array of shape [n_samples, n_targets]
            Target values
        threshold : the threshold for the coefficient below which it is discarded.
        n_components : the number of components to keep, after running PCA
        
        Returns
        -------
        self : returns an instance of self.
        """
        
        #these are the columns that will be removed
        self._leavouts=[]        
                
        dummy_X=X[:,np.newaxis]
        
        #test the absolute value of the coefficient for each variable. If it
        #is below a the threshold, then add it to the leaveouts        
        for i in range(0,dummy_X.shape[2]):
            current_X=dummy_X[:,:,i]
            self._model.fit(current_X, y)
            #the all([]) syntax is there in order to support both linear and logistic
            #regression. Logistic regression coefficients for multiclass problems
            #come in multi-dimensional arrays.
            if(all([abs(self._model.coef_[0])<self._threshold])):
                self._leavouts.append(i)
        
        if(len(self._leavouts)==dummy_X.shape[2]):
            raise ValueError('The total number of features to be left out is equal to the total number of features. Please try with a smaller threshold value.')

        
        #delete the variables that were below the threshold
        if(len(self._leavouts)>0):
            dummy_X=np.delete(dummy_X,self._leavouts,2)
        
        #conduct PCA for the designated number of components.
        #If no number was designated (or an illegal value<=0) then use the max number of component
        if(self._n_components>0):
            self._pca = PCA(n_components=self._n_components)
        else:
            self._pca = PCA(n_components=dummy_X.shape[2])
        dummy_X=self._pca.fit_transform(dummy_X[:,0,:])
        
        self._model=self._model.fit(dummy_X,y)
        
        return self
        
    def predict(self,X):
        """Predict using the supervised PCA model
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.
        Returns
        -------
        C : array, shape = (n_samples,)
            Returns predicted values.        
        """
        #remove the leavouts, transform the data and fit the regression model
        transformed_X=self.get_transformed_data(X)
        return self._model.predict(transformed_X)
    
    def get_transformed_data(self,X):
        """Calculates the components on a new matrix.
        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples,n_features]
            
        Returns
        -------
        transformed_X: Returns a transformed numpy array or sparse matrix. The
        leavouts have been removed and the remaining variables are transformed into
        components using the weights of the PCA object.
        
        Notes
        -------
        The algorithm should have first been executed on a dataset.
        
        """
        transformed_X=np.delete(X,self._leavouts,1)
        transformed_X=self._pca.transform(transformed_X)
        return transformed_X
        
    def get_n_components(self):
        return self._pca.n_components_
    
    
    #I am implementing a function here to get the components in order to avoid
    #the user having to access the pca object. Another option would be to 
    #copy the components from the pca to a variable located at 'self'. However,
    #this might be too redundant.
    def get_components(self):
        """Returns the components formerly calculated on a training dataset.
            
        Returns
        -------
        components: A numpy matrix with the loadings of the PCA components.
        
        Notes
        -------
        The algorithm should have first been executed on a dataset.
        
        """
        return self._pca.components_
    
    #same principle as in the get_components function
    def get_coefs(self):
        return self._model.coef_
        
    def score(self,X,y):
        return self._model.score(X,y)
    
        
        
class SupervisedPCARegressor(SupervisedPCA,RegressorMixin):
    """
    Implementation of supervisedPCA for regression. The underlying model
    is a linear regression model.
    
    Parameters
    ----------
    normalize : boolean, optional, default False
        If True, the regressors X will be normalized before regression.
    copy_X : boolean, optional, default True
        If True, X will be copied; else, it may be overwritten.
    n_jobs : int, optional, default 1
        The number of jobs to use for the computation.
        If -1 all CPUs are used. This will only provide speedup for
        n_targets > 1 and sufficient large problems.
    Attributes
    ----------
    coef_ : array, shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.
        If multiple targets are passed during the fit (y 2D), this
        is a 2D array of shape (n_targets, n_features), while if only
        one target is passed, this is a 1D array of length n_features.
    intercept_ : array
        Independent term in the linear model.
    
    """
    def __init__(self,fit_intercept=True, normalize=False, copy_X=True,n_jobs=1,threshold=0,n_components=-1):
        model=LinearRegression(copy_X=copy_X,normalize=normalize,n_jobs=n_jobs)  
        super(SupervisedPCARegressor,self).__init__(fit_intercept=fit_intercept,model=model,threshold=threshold,n_components=n_components)


class SupervisedPCAClassifier(SupervisedPCA,ClassifierMixin):
    """Implementation of supervisedPCA for classification. The underlying model
    is a logistic regression model.
    Parameters
    ----------
    penalty : str, 'l1' or 'l2'
        Used to specify the norm used in the penalization. The newton-cg and
        lbfgs solvers support only l2 penalties.
    dual : bool
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.
    C : float, optional (default=1.0)
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger
        regularization.
    fit_intercept : bool, default: True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added the decision function.
    intercept_scaling : float, default: 1
        Useful only if solver is liblinear.
        when self.fit_intercept is True, instance vector x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equals to
        intercept_scaling is appended to the instance vector.
        The intercept becomes intercept_scaling * synthetic feature weight
        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.
    class_weight : dict or 'balanced', optional
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``
    max_iter : int
        Useful only for the newton-cg and lbfgs solvers. Maximum number of
        iterations taken for the solvers to converge.
    random_state : int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator to use when
        shuffling the data.
    solver : {'newton-cg', 'lbfgs', 'liblinear'}
        Algorithm to use in the optimization problem.
    tol : float, optional
        Tolerance for stopping criteria.
    multi_class : str, {'ovr', 'multinomial'}
        Multiclass option can be either 'ovr' or 'multinomial'. If the option
        chosen is 'ovr', then a binary problem is fit for each label. Else
        the loss minimised is the multinomial loss fit across
        the entire probability distribution. Works only for the 'lbfgs'
        solver.
    verbose : int
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.
    """
    def __init__(self,fit_intercept=True, normalize=False, copy_X=True,penalty='l2', dual=False, tol=1e-4, C=1.0,
                 intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0,threshold=0,n_components=-1):
        model=LogisticRegression()  
        super(SupervisedPCAClassifier,self).__init__(fit_intercept=fit_intercept,model=model,threshold=threshold,n_components=n_components)
    
    def predict_proba(self,X):
        return self._model.predict_proba(X)

import numpy as np
from copy import deepcopy
import torch
from torch.backends import cudnn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import Identity

from nnunet.network_architecture.generic_UNet import Upsample
from nnunet.network_architecture.generic_modular_UNet import PlainConvUNetDecoder, get_default_network_config
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from torch import nn
from torch.optim import SGD


class BasicPreActResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, props, stride=None):
        """
        This is norm nonlin conv norm nonlin conv
        :param in_planes:
        :param out_planes:
        :param props:
        :param override_stride:
        """
        super().__init__()

        self.kernel_size = kernel_size
        props['conv_op_kwargs']['stride'] = 1

        self.stride = stride
        self.props = props
        self.out_planes = out_planes
        self.in_planes = in_planes

        if stride is not None:
            kwargs_conv1 = deepcopy(props['conv_op_kwargs'])
            kwargs_conv1['stride'] = stride
        else:
            kwargs_conv1 = props['conv_op_kwargs']

        self.norm1 = props['norm_op'](in_planes, **props['norm_op_kwargs'])
        self.nonlin1 = props['nonlin'](**props['nonlin_kwargs'])
        self.conv1 = props['conv_op'](in_planes, out_planes, kernel_size, padding=[(i - 1) // 2 for i in kernel_size],
                                      **kwargs_conv1)

        if props['dropout_op_kwargs']['p'] != 0:
            self.dropout = props['dropout_op'](**props['dropout_op_kwargs'])
        else:
            self.dropout = Identity()

        self.norm2 = props['norm_op'](out_planes, **props['norm_op_kwargs'])
        self.nonlin2 = props['nonlin'](**props['nonlin_kwargs'])
        self.conv2 = props['conv_op'](out_planes, out_planes, kernel_size, padding=[(i - 1) // 2 for i in kernel_size],
                                      **props['conv_op_kwargs'])

        if (self.stride is not None and any((i != 1 for i in self.stride))) or (in_planes != out_planes):
            stride_here = stride if stride is not None else 1
            self.downsample_skip = nn.Sequential(props['conv_op'](in_planes, out_planes, 1, stride_here, bias=False))
        else:
            self.downsample_skip = None

    def forward(self, x):
        residual = x

        out = self.nonlin1(self.norm1(x))

        if self.downsample_skip is not None:
            residual = self.downsample_skip(out)

        # norm nonlin conv
        out = self.conv1(out)

        out = self.dropout(out) # this does nothing if props['dropout_op_kwargs'] == 0

        # norm nonlin conv
        out = self.conv2(self.nonlin2(self.norm2(out)))

        out += residual

        return out


class PreActResidualLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, network_props, num_blocks, first_stride=None):
        super().__init__()

        network_props = deepcopy(network_props)  # network_props is a dict and mutable, so we deepcopy to be safe.

        self.convs = nn.Sequential(
            BasicPreActResidualBlock(input_channels, output_channels, kernel_size, network_props, first_stride),
            *[BasicPreActResidualBlock(output_channels, output_channels, kernel_size, network_props) for _ in
              range(num_blocks - 1)]
        )

    def forward(self, x):
        return self.convs(x)


class PreActResidualUNetEncoder(nn.Module):
    def __init__(self, input_channels, base_num_features, num_blocks_per_stage, feat_map_mul_on_downscale,
                 pool_op_kernel_sizes, conv_kernel_sizes, props, default_return_skips=True,
                 max_num_features=480, pool_type: str = 'conv'):
        """
        Following UNet building blocks can be added by utilizing the properties this class exposes (TODO)
        this one includes the bottleneck layer!
        :param input_channels:
        :param base_num_features:
        :param num_blocks_per_stage:
        :param feat_map_mul_on_downscale:
        :param pool_op_kernel_sizes:
        :param conv_kernel_sizes:
        :param props:
        """
        super(PreActResidualUNetEncoder, self).__init__()

        self.default_return_skips = default_return_skips
        self.props = props

        pool_op = self._handle_pool(pool_type)

        self.stages = []
        self.stage_output_features = []
        self.stage_pool_kernel_size = []
        self.stage_conv_op_kernel_size = []

        assert len(pool_op_kernel_sizes) == len(conv_kernel_sizes)

        num_stages = len(conv_kernel_sizes)

        if not isinstance(num_blocks_per_stage, (list, tuple)):
            num_blocks_per_stage = [num_blocks_per_stage] * num_stages
        else:
            assert len(num_blocks_per_stage) == num_stages

        self.num_blocks_per_stage = num_blocks_per_stage  # decoder may need this

        self.initial_conv = props['conv_op'](input_channels, base_num_features, 3, padding=1, **props['conv_op_kwargs'])

        current_input_features = base_num_features
        for stage in range(num_stages):
            current_output_features = min(base_num_features * feat_map_mul_on_downscale ** stage, max_num_features)
            current_kernel_size = conv_kernel_sizes[stage]

            current_pool_kernel_size = pool_op_kernel_sizes[stage]
            if pool_op is not None:
                pool_kernel_size_for_conv = [1 for i in current_pool_kernel_size]
            else:
                pool_kernel_size_for_conv = current_pool_kernel_size

            current_stage = PreActResidualLayer(current_input_features, current_output_features, current_kernel_size, props,
                                                self.num_blocks_per_stage[stage], pool_kernel_size_for_conv)
            if pool_op is not None:
                current_stage = nn.Sequential(pool_op(current_pool_kernel_size), current_stage)

            self.stages.append(current_stage)
            self.stage_output_features.append(current_output_features)
            self.stage_conv_op_kernel_size.append(current_kernel_size)
            self.stage_pool_kernel_size.append(current_pool_kernel_size)

            # update current_input_features
            current_input_features = current_output_features

        self.stages = nn.ModuleList(self.stages)
        self.output_features = current_input_features

    def _handle_pool(self, pool_type):
        assert pool_type in ['conv', 'avg', 'max']
        if pool_type == 'avg':
            if self.props['conv_op'] == nn.Conv2d:
                pool_op = nn.AvgPool2d
            elif self.props['conv_op'] == nn.Conv3d:
                pool_op = nn.AvgPool3d
            else:
                raise NotImplementedError
        elif pool_type == 'max':
            if self.props['conv_op'] == nn.Conv2d:
                pool_op = nn.MaxPool2d
            elif self.props['conv_op'] == nn.Conv3d:
                pool_op = nn.MaxPool3d
            else:
                raise NotImplementedError
        elif pool_type == 'conv':
            pool_op = None
        else:
            raise ValueError
        return pool_op
from keras_unet import TF
if TF:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        BatchNormalization,
        Conv2D,
        Conv2DTranspose,
        MaxPooling2D,
        Dropout,
        SpatialDropout2D,
        UpSampling2D,
        Input,
        concatenate,
        multiply,
        add,
        Activation,
    )
else:
    from keras.models import Model
    from keras.layers import (
        BatchNormalization,
        Conv2D,
        Conv2DTranspose,
        MaxPooling2D,
        Dropout,
        SpatialDropout2D,
        UpSampling2D,
        Input,
        concatenate,
        multiply,
        add,
        Activation,
    )


def upsample_conv(filters, kernel_size, strides, padding):
    return Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)


def upsample_simple(filters, kernel_size, strides, padding):
    return UpSampling2D(strides)


def attention_gate(inp_1, inp_2, n_intermediate_filters):
    """Attention gate. Compresses both inputs to n_intermediate_filters filters before processing.
       Implemented as proposed by Oktay et al. in their Attention U-net, see: https://arxiv.org/abs/1804.03999.
    """
    inp_1_conv = Conv2D(
        n_intermediate_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
    )(inp_1)
    inp_2_conv = Conv2D(
        n_intermediate_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
    )(inp_2)

    f = Activation("relu")(add([inp_1_conv, inp_2_conv]))
    g = Conv2D(
        filters=1,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
    )(f)
    h = Activation("sigmoid")(g)
    return multiply([inp_1, h])


def attention_concat(conv_below, skip_connection):
    """Performs concatenation of upsampled conv_below with attention gated version of skip-connection
    """
    below_filters = conv_below.get_shape().as_list()[-1]
    attention_across = attention_gate(skip_connection, conv_below, below_filters)
    return concatenate([conv_below, attention_across])


def conv2d_block(
    inputs,
    use_batch_norm=True,
    dropout=0.3,
    dropout_type="spatial",
    filters=16,
    kernel_size=(3, 3),
    activation="relu",
    kernel_initializer="he_normal",
    padding="same",
):

    if dropout_type == "spatial":
        DO = SpatialDropout2D
    elif dropout_type == "standard":
        DO = Dropout
    else:
        raise ValueError(
            f"dropout_type must be one of ['spatial', 'standard'], got {dropout_type}"
        )

    c = Conv2D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=not use_batch_norm,
    )(inputs)
    if use_batch_norm:
        c = BatchNormalization()(c)
    if dropout > 0.0:
        c = DO(dropout)(c)
    c = Conv2D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=not use_batch_norm,
    )(c)
    if use_batch_norm:
        c = BatchNormalization()(c)
    return c


def network(
    input_shape,
    num_classes=1,
    activation="relu",
    use_batch_norm=True,
    upsample_mode="deconv",  # 'deconv' or 'simple'
    dropout=0.3,
    dropout_change_per_layer=0.0,
    dropout_type="spatial",
    use_dropout_on_upsampling=False,
    use_attention=False,
    filters=16,
    num_layers=4,
    output_activation="sigmoid",
):  # 'sigmoid' or 'softmax'

    """
    Customisable UNet architecture (Ronneberger et al. 2015 [1]).

    Arguments:
    input_shape: 3D Tensor of shape (x, y, num_channels)

    num_classes (int): Unique classes in the output mask. Should be set to 1 for binary segmentation

    activation (str): A keras.activations.Activation to use. ReLu by default.

    use_batch_norm (bool): Whether to use Batch Normalisation across the channel axis between convolutional layers

    upsample_mode (one of "deconv" or "simple"): Whether to use transposed convolutions or simple upsampling in the decoder part

    dropout (float between 0. and 1.): Amount of dropout after the initial convolutional block. Set to 0. to turn Dropout off

    dropout_change_per_layer (float between 0. and 1.): Factor to add to the Dropout after each convolutional block

    dropout_type (one of "spatial" or "standard"): Type of Dropout to apply. Spatial is recommended for CNNs [2]

    use_dropout_on_upsampling (bool): Whether to use dropout in the decoder part of the network

    use_attention (bool): Whether to use an attention dynamic when concatenating with the skip-connection, implemented as proposed by Oktay et al. [3]

    filters (int): Convolutional filters in the initial convolutional block. Will be doubled every block

    num_layers (int): Number of total layers in the encoder not including the bottleneck layer

    output_activation (str): A keras.activations.Activation to use. Sigmoid by default for binary segmentation

    Returns:
    model (keras.models.Model): The built U-Net

    Raises:
    ValueError: If dropout_type is not one of "spatial" or "standard"


    [1]: https://arxiv.org/abs/1505.04597
    [2]: https://arxiv.org/pdf/1411.4280.pdf
    [3]: https://arxiv.org/abs/1804.03999

    """

    if upsample_mode == "deconv":
        upsample = upsample_conv
    else:
        upsample = upsample_simple

    # Build U-Net model
    inputs = Input(input_shape)
    x = inputs

    down_layers = []
    for l in range(num_layers):
        x = conv2d_block(
            inputs=x,
            filters=filters,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            dropout_type=dropout_type,
            activation=activation,
        )
        down_layers.append(x)
        x = MaxPooling2D((2, 2))(x)
        dropout += dropout_change_per_layer
        filters = filters * 2  # double the number of filters with each layer

    x = conv2d_block(
        inputs=x,
        filters=filters,
        use_batch_norm=use_batch_norm,
        dropout=dropout,
        dropout_type=dropout_type,
        activation=activation,
    )

    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0

    for conv in reversed(down_layers):
        filters //= 2  # decreasing number of filters with each layer
        dropout -= dropout_change_per_layer
        x = upsample(filters, (2, 2), strides=(2, 2), padding="same")(x)
        if use_attention:
            x = attention_concat(conv_below=x, skip_connection=conv)
        else:
            x = concatenate([x, conv])
        x = conv2d_block(
            inputs=x,
            filters=filters,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            dropout_type=dropout_type,
            activation=activation,
        )

    outputs = Conv2D(num_classes, (1, 1), activation=output_activation)(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model
