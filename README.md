# RNN-to-Transformers
```RNN```
A simple RNN has one input layer, one hidden layer, and one output layer. The input layer receives the input sequence, and the output layer produces the output sequence. The hidden layer has a recurrent connection that allows the network to maintain a memory of previous inputs.

The equations for a simple RNN are:

`Hidden state`: $ s_t = f(Ux_t + Ws_{t-1} + b) $
`Output state`: $ y_t = g(Vs_t + c) $
where s_t is the hidden state at time `t`, `x_t` is the input at time `t`, `y_t` is the output at time `t, U, W,` and V are weight matrices, b and c are bias vectors, f is an activation function applied element-wise, and g is a softmax function applied element-wise.

The `RNN_net` class in your code implements a simple RNN with one input layer, one hidden layer, and one output layer. The equations for this RNN are:

`Hidden state`: ``` s_i = sigma(Ux_i + Ws_{i-1} + b) ```
`Output state`: ```$ y_i= O(Vs_i + c) $```
where s_i is the hidden state at time `i`, `x_i` is the input at time `i`, `y_i` is the output at time `i, U, W,` and `V` are weight matrices, b and c are bias vectors, and sigma is a non-linear activation function.

Now let's see how the RNN_net class works.

The `__init__` function initializes the parameters of the network, including the weight matrices and the bias vectors. Specifically, it initializes the weight matrices U, W, and V and the bias vectors b and c. It also initializes the layers of the network: i2h is a linear layer that maps the concatenated input and previous hidden state to the current hidden state, i2o is a linear layer that maps the concatenated input and previous hidden state to the output, and softmax is a log-softmax function that produces the output probabilities.

The forward function takes the input input_ and the previous hidden state hidden as input, computes the current hidden state and output using the weight matrices, bias vectors, and activation functions, and returns the output and the current hidden state. Specifically, it concatenates the input and previous hidden state to form the input to the i2h and i2o layers, computes the new hidden state hidden using the i2h layer, computes the output output using the i2o layer, and applies the softmax function to the output to produce the output probabilities.

The init_hidden function initializes the previous hidden state to a tensor of zeros with the appropriate dimensions



## Encoder-Decoder Inference

- Encoder input : `torch.Size([6, 1, 27])`
    ```the size is (6, 1, 27), which means that the tensor has 3 dimensions:
    The first dimension has size 6, which likely corresponds to the sequence length or the number of time steps in the input.
    The second dimension has size 1, which likely corresponds to the batch size.
    The third dimension has size 27, which could correspond to the number of features in each input element, or the size of the input vocabulary if the       input is composed of one-hot encoded vectors.```

- Encoder output : `torch.Size([6, 1, 256])`
    ```the size is (6, 1, 256), which means that the tensor has 3 dimensions:

    The first dimension has size 6, which corresponds to the sequence length or the number of time steps in the output.
    The second dimension has size 1, which corresponds to the batch size. In this case, the output seems to correspond to a batch of size 1.
    The third dimension has size 256, which corresponds to the hidden size of the encoder. This means that each element in the output sequence is a vector of length 256, which encodes the hidden state of the encoder at that time step.`
- Encoder hidden ```torch.Size([1, 1, 256])```
   ```the size is (1, 1, 256), which means that  the tensor has 3 dimensions:

    The first dimension has size 1, which corresponds to the number of layers in the encoder. In this case, it seems that the encoder only has one layer.
    The second dimension has size 1, which corresponds to the batch size. In this case, the output seems to correspond to a batch of size 1.
    The third dimension has size 256, which corresponds to the hidden size of the encoder. This means that the hidden state of the encoder is a vector of length 256.
    In other words, the encoder hidden state is a vector that summarizes the information in the input sequence that the encoder has seen so far. This hidden state is updated at each time step of the encoder, and can be passed to the decoder to help it generate the output sequence.

    Note that because the encoder hidden state is a vector of fixed length (256 in this case), it can be thought of as a bottleneck that compresses the information in the input sequence into a smaller representation that can be used by the decoder.```
- Decoder state ```torch.Size([1, 1, 256])```
    
    ```the size is (1, 1, 256), which means that the tensor has 3 dimensions:

    The first dimension has size 1, which corresponds to the number of layers in the decoder. In this case, it seems that the decoder only has one layer.
    The second dimension has size 1, which corresponds to the batch size. In this case, the output seems to correspond to a batch of size 1.
    The third dimension has size 256, which corresponds to the hidden size of the decoder. This means that the decoder state is a vector of length 256.
    In other words, the decoder state is a vector that summarizes the information in the output sequence that the decoder has generated so far. This hidden state is updated at each time step of the decoder, and can be used to generate the next output element in the sequence.

    Note that because the decoder state is a vector of fixed length (256 in this case), it can be thought of as a bottleneck that compresses the information in the output sequence into a smaller representation that can be used to generate subsequent output elements.```
- Decoder input ```torch.Size([1, 1, 129])```
    
    ```the size is (1, 1, 129), which means that the tensor has 3 dimensions:

    The first dimension has size 1, which corresponds to the sequence length or the number of time steps in the input. In this case, it seems that the decoder input has only one time step.
    The second dimension has size 1, which corresponds to the batch size. In this case, the output seems to correspond to a batch of size 1.
    The third dimension has size 129, which corresponds to the size of the vocabulary or the number of possible output elements that the decoder can generate. This means that the decoder input is a one-hot encoded vector of length 129, where the element corresponding to the next output element in the sequence is set to 1, and all other elements are set to 0.
    In other words, the decoder input is a one-hot encoded representation of the next element in the output sequence that the decoder needs to generate. This input is passed to the decoder at each time step, and is used along with the decoder state to generate the next output element. Note that because the decoder input is a one-hot encoded vector, it can be thought of as a sparse representation of the output sequence.```
- Decoder intermediate output ```torch.Size([1, 1, 256])```
    ```the size is (1, 1, 256), which means that the tensor has 3 dimensions:

    The first dimension has size 1, which corresponds to the sequence length or the number of time steps in the output. In this case, it seems that the decoder intermediate output has only one time step.
    The second dimension has size 1, which corresponds to the batch size. In this case, the output seems to correspond to a batch of size 1.
    The third dimension has size 256, which corresponds to the hidden size of the decoder. This means that the decoder intermediate output is a vector of length 256, which represents the hidden state of the decoder at the current time step.
    In other words, the decoder intermediate output is the output of the decoder at a particular time step, after processing the decoder input and the decoder state. This output is used to generate the next element in the output sequence. Note that because the decoder intermediate output is a vector of fixed length (256 in this case), it can be thought of as a bottleneck that compresses the information in the decoder input and the decoder state into a smaller representation that can be used to generate the next output element.```
- Decoder output ```torch.Size([1, 1, 129])```
    ```the size is (1, 1, 129), which means that the tensor has 3 dimensions:
    The first dimension has size 1, which corresponds to the sequence length or the number of time steps in the output. In this case, it seems that the decoder output has only one time step.
    The second dimension has size 1, which corresponds to the batch size. In this case, the output seems to correspond to a batch of size 1.
    The third dimension has size 129, which corresponds to the size of the vocabulary or the number of possible output elements that the decoder can generate. This means that the decoder output is a vector of length 129, where each element represents the probability of generating a particular output element (e.g., a particular word in a sequence) at the current time step.
    In other words, the decoder output is the final output of the decoder at a particular time step, which represents the probability distribution over the possible output elements that the decoder can generate. This output is typically used to select the most likely output element (e.g., the word with the highest probability) and add it to the output sequence. Note that because the decoder output is a probability distribution, it can be thought of as a dense representation of the output sequence, where each element corresponds to the probability of generating a particular output element.```

