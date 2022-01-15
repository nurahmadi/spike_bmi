"""
List of BMI decoders which consist of classical and deep learning algorithms.
1. Kalman filter (KF) 
2. Wiener filter (WF)
3. Multilayer perceptron (MLP)
4. Long short-term memory (LSTM)
5. Quasi-recurrent neural network (QRNN)

"""

# import packages
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from numpy.linalg import inv, pinv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Layer, InputSpec
from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.python.keras.utils.conv_utils import conv_output_length
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K

class KalmanDecoder:
    """
    Kalman filter decoding algorithm.

    Parameters
    ----------
        reg_type : str {'l1', 'l2', 'l12'}. Default None.
            regularization type.
            'l1' : Linear least squares with l1 regularization (i.e. Lasso).
            'l2' : Linear least squares with l2 regularization (i.e. Ridge).
            'l12' : Linear least squares with combined L1 and L2 regularization (i.e. Elastic Net).
        reg_alpha : float. Default 0.
            regularization constant/strength.
    """
    def __init__(self, reg_type=None, reg_alpha=0):
        self.reg_type = reg_type 
        self.reg_alpha = reg_alpha

    def fit(self, X_train, y_train):
        if self.reg_type == 'l1':
            regres = Lasso(alpha=self.alpha_reg)            
        elif self.reg_type == 'l2':
            regres = Ridge(alpha=self.alpha_reg)
        elif self.reg_type == 'l12':
            regres = ElasticNet(alpha=self.alpha_reg)
        else:
            regres = LinearRegression()
        
        X = y_train 
        Z = X_train 
        nt = X.shape[0]              
        X1 = X[:nt-1,:] 
        X2 = X[1:,:] 
        
        regres.fit(X1, X2)
        A = regres.coef_
        W = np.cov((X2 - np.dot(X1, A.T)).T)
        regres.fit(X, Z)
        H = regres.coef_ 
        Q = np.cov((Z - np.dot(X, H.T)).T) 
        self.model = [A, W, H, Q] 
    
    def predict(self, X_test, y_init):
        # extract parameters
        A, W, H, Q = self.model

        X = np.matrix(y_init.T)
        Z = np.matrix(X_test.T)

        # initialise states and covariance matrix
        n_states = X.shape[0] # dimensionality of the state
        states = np.empty((n_states, Z.shape[1])) # keep track of states over time (states is what will be returned as y_pred)
        P_m = np.matrix(np.zeros([n_states, n_states]))
        P = np.matrix(np.zeros([n_states, n_states]))
        state = X[:,0] # initial state
        states[:,0] = np.copy(np.squeeze(state))

        # get predicted state for every time bin
        for t in range(Z.shape[1]-1):
            # do first part of state update - based on transition matrix
            P_m = A*P*A.T + W
            state_m = A*state

            # do second part of state update - based on measurement matrix
            try:
                K = P_m*H.T*inv(H*P_m*H.T + Q) # calculate Kalman gain
            except np.linalg.LinAlgError:
                K = P_m*H.T*pinv(H*P_m*H.T+Q) # calculate Kalman gain
            P = (np.matrix(np.eye(n_states)) - K*H)*P_m
            state = state_m + K*(Z[:,t+1] - H*state_m)
            states[:,t+1] = np.squeeze(state) # record state at the timestep
        y_pred = states.T
        return y_pred

class WienerDecoder:
    """
    Wiener filter decoding algorithm.

    Parameters
    ----------
        reg_type : str {'l1', 'l2', 'l12'}. Default None.
            regularization type.
            'l1' : Linear least squares with l1 regularization (i.e. Lasso).
            'l2' : Linear least squares with l2 regularization (i.e. Ridge).
            'l12' : Linear least squares with combined L1 and L2 regularization (i.e. Elastic Net).
        reg_alpha : float. Default 0.
            regularization constant/strength.
    """
    def __init__(self, reg_type=None, reg_alpha=0):
        self.reg_type = reg_type
        self.reg_alpha = reg_alpha

    def fit(self, X_train, y_train):
        if self.reg_type == 'l1':
            self.model = Lasso(alpha=self.reg_alpha)            
        elif self.reg_type == 'l2':
            self.model = Ridge(alpha=self.reg_alpha)
        elif self.reg_type == 'l12':
            self.model = ElasticNet(alpha=self.reg_alpha)
        else:
            self.model = LinearRegression()
        
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test) # make predictions
        return y_pred

def MLPDecoder(config):
    """
    Multilayer perceptron (MLP) decoding algorithm.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input.
    output_dim : int
        Dimensionality of the output.
    n_layers : int
        Number of layers.
    units : int
        Dimensionality of the output space.
    dropout : float, between 0 and 1
        Fraction of the input units to drop.
    learning_rate : float
        The learning rate.
    optimizer : str, {'RMSprop', 'Adam'}
        Name of the optimizer.
    loss : str
        Loss function.
    metric : str
        Metric to be evaluated.

    Return
    ----------
    model : class
        Model class instance.
    """
    model = Sequential()
    for i in range(config["n_layers"]):
        if i==0:
            model.add(Dense(units=config["units"], input_shape=(config["input_dim"],), activation='relu'))
        else:
            model.add(Dense(units=config["units"], activation='relu'))
        model.add(Dropout(config["dropout"]))
    model.add(Dense(units=config["output_dim"]))

    if config["optimizer"].lower() == "rmsprop":
        opt = RMSprop(learning_rate=config["learning_rate"])
    else:
        opt = Adam(learning_rate=config["learning_rate"])

    model.compile(loss=config["loss"],
                  optimizer=opt,
                  metrics=config["metric"])
    #print(model.summary())
    return model

def LSTMDecoder(config):
    """
    Long short-term memory (LSTM) decoding algorithm.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input.
    output_dim : int
        Dimensionality of the output.
    n_layers : int
        Number of layers.
    units : int
        Dimensionality of the output space.
    timesteps : int
        Number of timesteps.
    dropout : float, between 0 and 1
        Fraction of the input units to drop.
    learning_rate : float
        The learning rate.
    optimizer : str, {'RMSprop', 'Adam'}
        Name of the optimizer.
    loss : str
        Loss function.
    metric : str
        Metric to be evaluated.

    Return
    ----------
    model : class
        Model class instance.
    """
    model = Sequential()
    for i in range(config["n_layers"]):
        if i==config["n_layers"]-1:
            model.add(LSTM(config["units"], input_shape=(config["timesteps"], config["input_dim"]), return_sequences=False))
        else:
            model.add(LSTM(config["units"], input_shape=(config["timesteps"], config["input_dim"]), return_sequences=True))
    if config["dropout"] > 0:
        model.add(Dropout(config["dropout"]))
    model.add(Dense(units=config["output_dim"]))
    
    if config["optimizer"].lower() == "rmsprop":
        opt = RMSprop(learning_rate=config["learning_rate"])
    else:
        opt = Adam(learning_rate=config["learning_rate"])

    model.compile(loss=config["loss"],
                  optimizer=opt,
                  metrics=config["metric"])
    #print(model.summary())
    return model

def _dropout(x, level, noise_shape=None, seed=None):
    x = K.dropout(x, level, noise_shape, seed)
    x *= (1. - level) # compensate for the scaling by the dropout
    return x

class QRNN(Layer):
    '''Quasi RNN
    # Arguments
        units: dimension of the internal projections and the final output.
    # References
        - [Quasi-recurrent Neural Networks](http://arxiv.org/abs/1611.01576)
    source: https://github.com/DingKe/nn_playground/blob/master/qrnn/qrnn.py
    '''
    def __init__(self, units, window_size=2, stride=1,
                 return_sequences=False, go_backwards=False, 
                 stateful=False, unroll=False, activation='tanh',
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, 
                 dropout=0, use_bias=True, input_dim=None, input_length=None,
                 **kwargs):
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll

        self.units = units 
        self.window_size = window_size
        self.strides = (stride, 1)

        self.use_bias = use_bias
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = dropout
        self.supports_masking = True
        self.input_spec = [InputSpec(ndim=3)]
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(QRNN, self).__init__(**kwargs)

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        self.input_dim = input_shape[2]
        self.input_spec = InputSpec(shape=(batch_size, None, self.input_dim))
        self.state_spec = InputSpec(shape=(batch_size, self.units))

        self.states = [None]
        if self.stateful:
            self.reset_states()

        kernel_shape = (self.window_size, 1, self.input_dim, self.units * 3)
        self.kernel = self.add_weight(name='kernel',
                                      shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(name='bias', 
                                        shape=(self.units * 3,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)

        self.built = True

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        length = input_shape[1]
        if length:
            length = conv_output_length(length + self.window_size - 1,
                                        self.window_size, 'valid',
                                        self.strides[0])
        if self.return_sequences:
            return (input_shape[0], length, self.units)
        else:
            return (input_shape[0], self.units)

    def compute_mask(self, inputs, mask):
        if self.return_sequences:
            return mask
        else:
            return None

    def get_initial_states(self, inputs):
        # build an all-zero tensor of shape (samples, units)
        initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        initial_state = K.tile(initial_state, [1, self.units])  # (samples, units)
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states

    def reset_states(self, states=None):
        if not self.stateful:
            raise AttributeError('Layer must be stateful.')
        if not self.input_spec:
            raise RuntimeError('Layer has never been called '
                               'and thus has no states.')

        batch_size = self.input_spec.shape[0]
        if not batch_size:
            raise ValueError('If a QRNN is stateful, it needs to know '
                             'its batch size. Specify the batch size '
                             'of your input tensors: \n'
                             '- If using a Sequential model, '
                             'specify the batch size by passing '
                             'a `batch_input_shape` '
                             'argument to your first layer.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a '
                             '`batch_shape` argument to your Input layer.')

        if self.states[0] is None:
            self.states = [K.zeros((batch_size, self.units))
                           for _ in self.states]
        elif states is None:
            for state in self.states:
                K.set_value(state, np.zeros((batch_size, self.units)))
        else:
            if not isinstance(states, (list, tuple)):
                states = [states]
            if len(states) != len(self.states):
                raise ValueError('Layer ' + self.name + ' expects ' +
                                 str(len(self.states)) + ' states, '
                                 'but it received ' + str(len(states)) +
                                 'state values. Input received: ' +
                                 str(states))
            for index, (value, state) in enumerate(zip(states, self.states)):
                if value.shape != (batch_size, self.units):
                    raise ValueError('State ' + str(index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected shape=' +
                                     str((batch_size, self.units)) +
                                     ', found shape=' + str(value.shape))
                K.set_value(state, value)

    def __call__(self, inputs, initial_state=None, **kwargs):
        # If `initial_state` is specified,
        # and if it a Keras tensor,
        # then add it to the inputs and temporarily
        # modify the input spec to include the state.
        if initial_state is not None:
            if hasattr(initial_state, '_keras_history'):
                # Compute the full input spec, including state
                input_spec = self.input_spec
                state_spec = self.state_spec
                if not isinstance(state_spec, list):
                    state_spec = [state_spec]
                self.input_spec = [input_spec] + state_spec

                # Compute the full inputs, including state
                if not isinstance(initial_state, (list, tuple)):
                    initial_state = [initial_state]
                inputs = [inputs] + list(initial_state)

                # Perform the call
                output = super(QRNN, self).__call__(inputs, **kwargs)

                # Restore original input spec
                self.input_spec = input_spec
                return output
            else:
                kwargs['initial_state'] = initial_state
        return super(QRNN, self).__call__(inputs, **kwargs)

    def call(self, inputs, mask=None, initial_state=None, training=None):
        # input shape: `(samples, time (padded with zeros), input_dim)`
        # note that the .build() method of subclasses MUST define
        # self.input_spec and self.state_spec with complete input shapes.
        if isinstance(inputs, list):
            initial_states = inputs[1:]
            inputs = inputs[0]
        elif initial_state is not None:
            pass
        elif self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(inputs)

        if len(initial_states) != len(self.states):
            raise ValueError('Layer has ' + str(len(self.states)) +
                             ' states but was passed ' +
                             str(len(initial_states)) +
                             ' initial states.')
        input_shape = K.int_shape(inputs)
        if self.unroll and input_shape[1] is None:
            raise ValueError('Cannot unroll a RNN if the '
                             'time dimension is undefined. \n'
                             '- If using a Sequential model, '
                             'specify the time dimension by passing '
                             'an `input_shape` or `batch_input_shape` '
                             'argument to your first layer. If your '
                             'first layer is an Embedding, you can '
                             'also use the `input_length` argument.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a `shape` '
                             'or `batch_shape` argument to your Input layer.')
        constants = self.get_constants(inputs, training=None)
        preprocessed_input = self.preprocess_input(inputs, training=None)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                            initial_states,
                                            go_backwards=self.go_backwards,
                                            mask=mask,
                                            constants=constants,
                                            unroll=self.unroll,
                                            input_length=input_shape[1])
        if self.stateful:
            updates = []
            for i in range(len(states)):
                updates.append((self.states[i], states[i]))
            self.add_update(updates, inputs)

        # Properly set learning phase
        if 0 < self.dropout < 1:
            last_output._uses_learning_phase = True
            outputs._uses_learning_phase = True

        if self.return_sequences:
            return outputs
        else:
            return last_output

    def preprocess_input(self, inputs, training=None):
        if self.window_size > 1:
            inputs = K.temporal_padding(inputs, (self.window_size-1, 0))
        inputs = K.expand_dims(inputs, 2)  # add a dummy dimension

        output = K.conv2d(inputs, self.kernel, strides=self.strides,
                          padding='valid',
                          data_format='channels_last')
        output = K.squeeze(output, 2)  # remove the dummy dimension
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')

        if self.dropout is not None and 0. < self.dropout < 1.:
            z = output[:, :, :self.units]
            f = output[:, :, self.units:2 * self.units]
            o = output[:, :, 2 * self.units:]
            f = K.in_train_phase(1 - _dropout(1 - f, self.dropout), f, training=training)
            return K.concatenate([z, f, o], -1)
        else:
            return output

    def step(self, inputs, states):
        prev_output = states[0]

        z = inputs[:, :self.units]
        f = inputs[:, self.units:2 * self.units]
        o = inputs[:, 2 * self.units:]

        z = self.activation(z)
        f = f if self.dropout is not None and 0. < self.dropout < 1. else K.sigmoid(f)
        o = K.sigmoid(o)

        #output = f * prev_output + (1 - f) * z
        #output = o * output
        c_output = f * prev_output + (1 - f) * z
        h_output = o * c_output

        #return output, [output]
        return h_output, [c_output]

    def get_constants(self, inputs, training=None):
        return []
 
    def get_config(self):
        config = {'units': self.units,
                  'window_size': self.window_size,
                  'stride': self.strides[0],
                  'return_sequences': self.return_sequences,
                  'go_backwards': self.go_backwards,
                  'stateful': self.stateful,
                  'unroll': self.unroll,
                  'use_bias': self.use_bias,
                  'dropout': self.dropout,
                  'activation': activations.serialize(self.activation),
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'input_dim': self.input_dim,
                  'input_length': self.input_length}
        base_config = super(QRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def QRNNDecoder(config):
    """
    Long short-term memory (LSTM) decoding algorithm.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input.
    output_dim : int
        Dimensionality of the output.
    n_layers : int
        Number of layers.
    units : int
        Dimensionality of the output space.
    window_size : int
        Window size.
    dropout : float, between 0 and 1
        Fraction of the input units to drop.
    learning_rate : float
        The learning rate.
    optimizer : str, {'RMSprop', 'Adam'}
        Name of the optimizer.
    loss : str
        Loss function.
    metric : str
        Metric to be evaluated.

    Return
    ----------
    model : class
        Model class instance.
    """
    model = Sequential()
    for i in range(config["n_layers"]):
        if i==config["n_layers"]-1:
            model.add(QRNN(config["units"], input_shape=(config["timesteps"], config["input_dim"]), window_size=config["window_size"], return_sequences=False))
        else:
            model.add(QRNN(config["units"], input_shape=(config["timesteps"], config["input_dim"]), window_size=config["window_size"], return_sequences=True))
    if config["dropout"] > 0:
        model.add(Dropout(config["dropout"]))
    model.add(Dense(units=config["output_dim"]))
    
    if config["optimizer"].lower() == "rmsprop":
        opt = RMSprop(learning_rate=config["learning_rate"])
    else:
        opt = Adam(learning_rate=config["learning_rate"])

    model.compile(loss=config["loss"],
                  optimizer=opt,
                  metrics=config["metric"])
    #print(model.summary())
    return model
        