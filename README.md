# RNN-to-Transformers
```RNN```
A simple RNN has one input layer, one hidden layer, and one output layer. The input layer receives the input sequence, and the output layer produces the output sequence. The hidden layer has a recurrent connection that allows the network to maintain a memory of previous inputs.

The equations for a simple RNN are:

`Hidden state`: $ s_t = f(Ux_t + Ws_{t-1} + b) $
`Output state`: $ y_t = g(Vs_t + c) $
where s_t is the hidden state at time `t`, `x_t` is the input at time `t`, `y_t` is the output at time `t, U, W,` and V are weight matrices, b and c are bias vectors, f is an activation function applied element-wise, and g is a softmax function applied element-wise.

The `RNN_net` class in your code implements a simple RNN with one input layer, one hidden layer, and one output layer. The equations for this RNN are:

`Hidden state`: $ s_i = sigma(Ux_i + Ws_{i-1} + b) $
`Output state`: $ y_i= O(Vs_i + c) $
where s_i is the hidden state at time `i`, `x_i` is the input at time `i`, `y_i` is the output at time `i, U, W,` and `V` are weight matrices, b and c are bias vectors, and sigma is a non-linear activation function.

Now let's see how the RNN_net class works.

The `__init__` function initializes the parameters of the network, including the weight matrices and the bias vectors. Specifically, it initializes the weight matrices U, W, and V and the bias vectors b and c. It also initializes the layers of the network: i2h is a linear layer that maps the concatenated input and previous hidden state to the current hidden state, i2o is a linear layer that maps the concatenated input and previous hidden state to the output, and softmax is a log-softmax function that produces the output probabilities.

The forward function takes the input input_ and the previous hidden state hidden as input, computes the current hidden state and output using the weight matrices, bias vectors, and activation functions, and returns the output and the current hidden state. Specifically, it concatenates the input and previous hidden state to form the input to the i2h and i2o layers, computes the new hidden state hidden using the i2h layer, computes the output output using the i2o layer, and applies the softmax function to the output to produce the output probabilities.

The init_hidden function initializes the previous hidden state to a tensor of zeros with the appropriate dimensions.
