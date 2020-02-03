---
layout:     post
title:       Decoding convolutional codes 
date:       2020-02-01 12:31:19
summary:    Learning Viterbi Maximum Likelihood decoders for convolutional codes
categories: jekyll pixyll
comments:   true
---

--[*Hyeji Kim*](http://sites.utexas.edu/hkim/) and [*Sewoong Oh*](https://homes.cs.washington.edu/~sewoong/)

Viterbi algorithm exactly computes the Maximum Likelihood (ML) estimate of the transmitted codeword, 
by efficiently running a dynamic programming. 
Although statistically this cannot be improved upon, 
we train a deep neural network to re-discover a Viterbi-like *algorithm* that matches the ML performance. 
Here the emphasis is on "algorithm"; we want to learn a decoder 
that can be readily applied as is to any block lengths, beyond what it was trained on. 
In particular, a trained decoder is not considered an algorithm, if it only works for a fixed block length.

The purpose of this exercise is twofold. 
First, our ultimate goal is the discovery of new codes (encoders and decoders). 
Demonstrating that deep learning can reproduce the optimal decoder for existing codes is a necessary intermediate step. 
Next, we might want to impose additional constraints on the decoder, such as low latency. 
Deep learning provides such flexibility  which Viterbi algorithm is not equipped with. 

Consider communicating a message over a noisy channel.  This communication system has an encoder that maps messages (e.g., bit sequences) to codewords, typically of longer lengths, and a decoder that maps noisy codewords to the estimate of messages. This is illustrated below. 

<center><img src="https://deepcomm.github.io/images/commsystem.png" width="750"/></center>

Among the variety of existing codes, we choose a simple family of codes 
known as covolutional codes. Several properties make them ideal for our experiement. 

* These codes are practical. They are used for mobile communications (e.g., 4G LTE) and satellite communications. 
  
* These codes achieve performance close to the fundamental limit.
  
* The recurrent nature of sequential encoding aligns very well with the Recurrent Neural Network (RNN) structure. 
  
* Well-known decoders exist for these codes. For convolutional codes, maximum likelihood decoder on AWGN channels is Viterbi decoder, which is a dynamic programming. 
<!--For turbo codes, a belief propagation decoder on AWGN channels achieve performance close to the theoretical (Shannon) limit. Hence, learning a decoder for sequential codes poses the challenge of *learning an algorithm.*
-->

<!--
## Sequential code

When we fix the encoder, among many standard codes, we choose sequential codes such as *convolutional codes* and *turbo codes* for the following reasons:

* These codes are practical. They are used for mobile communications (e.g., 4G LTE) and satellite communications. 
  
* These codes achieve performance close to the fundamental limit.
  
* The recurrent nature of sequential encoding aligns very well with the Recurrent Neural Network (RNN) structure. 
  
* Well-known decoders exist for these codes. For convolutional codes, maximum likelihood decoder on AWGN channels is Viterbi decoder, which is a dynamic programming. For turbo codes, a belief propagation decoder on AWGN channels achieve performance close to the theoretical (Shannon) limit. Hence, learning a decoder for sequential codes poses the challenge of *learning an algorithm.*
-->




<!--Turbo code is an extension of convolutional codes developed in 1991 by [Claude Berrou](https://en.wikipedia.org/wiki/Claude_Berrou). In this post, we focus on decoding of convolutional codes; we will look into turbo codes in the next post. 
-->

### Convolutional codes

Convolutional codes are introduced in 1955 by [Peter Elias](https://en.wikipedia.org/wiki/Peter_Elias). 
It uses a short memory and connvolution operators to sequentially create coded bits. 
An example for a rate 1/2 convolutional code is shown below. This code maps  b<sub>k</sub> to  (c<sub>k1</sub>, c<sub>k2</sub>), where the state is  (b<sub>k</sub>,  b<sub>k-1</sub>,  b<sub>k-2</sub>), and coded bits (c<sub>k1</sub>, c<sub>k2</sub>) are convolution (i.e., mod 2 sum) of the state bits.  

<center><img src="https://hyejikim1.github.io/images/convcode.png"></center>





### Viterbi decoding

Around a decade after convolutional codes were introduced, in 1967, Andrew Viterbi came up with Viterbi algorithm, which is a dynamic programming algorithm for finding the most likely sequence of hidden states given an observed sequence in hidden Markov Models (HMM)s. Viterbi algorithm can find the maximum likelihood message bit sequence **b**  given the received signal **y** = **c** + **n**,  in a computationally efficient manner. We give an overview of Viterbi decoding. For a detailed walkthrough, nice tutorials can be found [here](https://web.stanford.edu/~jurafsky/slp3/A.pdf) and [here](https://www.researchgate.net/publication/31595950_Asynchronous_Viterbi_Decoder_in_Action_Systems). 

Convolutional codes can be seen as a state-transition diagram. The state diagram of the rate 1/2 convolutional code introduced above is as follows ([figure credit](https://www.researchgate.net/publication/31595950_Asynchronous_Viterbi_Decoder_in_Action_Systems)). States are depicted as nodes. An arrow with (b<sub>k</sub>/c<sub>k</sub>) from s<sub>k</sub> to s<sub>k+1</sub> represents a transition caused by b<sub>k</sub> on s<sub>k</sub>; coded bits ck are generated and next state is s<sub>k+1</sub>. 





<center><img src="https://www.researchgate.net/profile/Juha_Plosila/publication/31595950/figure/fig3/AS:654072529039362@1532954451991/State-diagram-for-rate-1-2-K3-convolutional-code_W640.jpg"></center>



<!---State - four possible options - evolve through time.  State to state transition is occured by new input bk. Each transition is assigned the specific transmitted codeword: ck is a function of (bk,sk) Cost is assigned to each transition (sk, bk, sk+1). Cost is how likely to observe the received coded bits ck + nk --->

<br />

Trellis diagram unrolls the transition across time. Let s<sub>0</sub>, s<sub>1</sub>, s<sub>2</sub>, s<sub>3</sub> denote the state (00),(01),(10),(11). All possible transitions rare marked with arrows, where solid line implies input b<sub>k</sub> is 0, and dashed line implies input b<sub>k</sub> is 1.  Coded bits b<sub>k</sub> are accompanied with each arrow. 

<br /> 

<center><img src="https://www.researchgate.net/profile/Juha_Plosila/publication/31595950/figure/fig4/AS:654072533221376@1532954452005/Trellis-diagram-for-rate-1-2-K3-convolutional-code_W640.jpg"></center>





<br /> 

In the trellis diagram, for each transition, let L<sub>k</sub> denote the likelihood, defined as the likelihood of  observing ***y<sub>k</sub>*** given the ground truth codeword is ***c<sub>k</sub>*** (e.g., 00,01,10,11). We aim to find a path (sequence of arrows) that maximizes the sum of L<sub>k</sub> for k=1 to K. 

<!---For t=1,2,3, ... we compute the best accumulated likelihood we can get conditioned on that we are at state sk at time k. --->

The high-level idea is as follows. Suppose we know the most likely path to get to state s<sub>k</sub> (0,1,2,3) at time k and the the corresponding sum of  L<sub>k</sub> for k = 1 to K. Given this, we can compute the most likely path to get to state s<sub>k+1</sub> (0,1,2,3) at time k+1 as follows: we take max<sub>s<sub>k</sub> in {0,1,2,3} </sub> (Accumulated likelihood at  s<sub>k</sub> at time k + likelihood due to the transition from s<sub>k</sub> to s<sub>k +1</sub>). We record the path (input) as well as the updated likelihood sum. After going through this process until k reaches K, we find s<sub>K</sub> that has the maximum accumulatd likelihood. The saved path to s<sub>k</sub> is enough to find the most likely input sequence ***b***.   



### Viterbi decoding as a neural network 

<center><img src="https://deepcomm.github.io/images/learndec.png" width="750"/></center>

It is also well known that recurrent neural networks can in principle implement any algorithm [Siegelmann and Sontag, 1992](https://ieeexplore.ieee.org/document/531522). Indeed, in 1996, [Wang and Wicker](https://ieeexplore.ieee.org/document/531522) has shown that artificial neural networks with hand-picked operations can emulate Viterbi decoder.

<!-- It took more than one decade for Viterbi decoder to be discovered since convolutional codes were introduced. The answer is, surprisingly, yes!  
-->

What is not clear and challenging is whether we can *learn* this decoder in a data-driven manner. We explain in this note how to learn a neural network based decoder for convolutional codes. We will see that its reliability matches that of Viterbi algorithm, across various SNRs and code lengths. Full code can be accessed [here](https://github.com/deepcomm/RNNViterbi). 






<!-- The design of channel codes is directly related to the reliability of communication systems; a practical value of reliable codes is enormous. The design of codes is also theoretically challenging and interesting; it has been a major area of study in information theory and coding theory for several decades since Shannon's 1948 seminal paper. 
-->

<!--
As a first step towards revolutionizing channel coding via deep learning (e.g., learning a novel pair of encoder-decoder), we ask the very first natural question: **Can we learn an optimal decoder for a fixed encoder from data?**  To answer this question, we fix the encoder as one of the standard encoders, model the decoder as a neural network, and train the decoder in a supervised manner.
-->







### Connection between sequential codes and RNNs

Below is an illustration of a *sequential* code that maps a message bit sequence **b** of length K to a codeword sequence **c**. The encoder first takes the first bit b<sub>1</sub>, update the state s<sub>1</sub>, and the generate coded bits **c<sub>1</sub>** based on the state s<sub>1</sub>. The encoder then takes the second bit b<sub>2</sub>, generate state s<sub>2</sub> based on (s<sub>1</sub>, b<sub>2</sub>), and then generate coded bits **c<sub>2</sub>**. These mappings occur recurrently until the last coded bits **c<sub>k</sub>** are generated. Each coded bits **c<sub>k</sub>** (k=1, ... K) is of length r when code rate is 1/r. For example, for code rate 1/2, **c<sub>1</sub>** is of length 2. 



<center><img src="https://deepcomm.github.io/images/seqcode.png"></center>







Recurrent Neural Network (RNN) is a neural architecture that's well suited for sequential mappings with memory. There is a hidden state h evolving through time. The hidden state keeps information on the current and all the past inputs. The hidden state is updated as a function (f) of previous hidden state and the input at each time. The output then is another function (g) of the hidden state at time k. In RNN, these f and g are parametric functions. Depending on what parametric functions we choose, the RNN can be a vanilla RNN, or LSTM, or GRU. See [here](http://dprogrammer.org/rnn-lstm-gru) for a detailed description. Once we choose the parametric function, we then learn  good parameters through training. So the RNN is a very natural fit to the sequential encoders. 

<center><img src="https://deepcomm.github.io/images/RNN.png"></center>





## Learning an RNN decoder for convolutional codes

Learning a decoder is very simple. It is done in four steps. 

* Step 1. Create a neural network model 
  
* Step 2. Choose an optimizer, a loss function, and an evaluation metric
  
* Step 3. Generate training examples and train the model 
  
* Step 4. Test the trained model on various test examples 



### Step 1. Modelling the decoder as an RNN

The first thing to do is to model the decoder as a neural network. We model the decoder as a Bi-directional RNN that has a forward pass and a backward pass. This is because we would like the decoder to estimate each bit based on the whole received sequence. In particular, we use GRU; empirically, we see GRU and LSTM have similar performance. We also use two layers of Bi-GRU because The input to each 1st layer GRU cell is received coded bits, i.e., noise sequence **n<sub>k</sub>**  added to the k-th coded bits **c<sub>k</sub>**. The output of each 2nd layer GRU cell is the estimate of b<sub>k</sub>. 



<center><img src="https://hyejikim1.github.io/images/twolayerbiGRUDec.png"></center>

Here is an excerpt of python code that defines the decoder neural network. In this post, we introduce codes built on Keras library, which is arguably one of the easiest deep learning libraries, as a gentle introduction to deep learning programming for channel coding.  The complete code and installation guides can be found [here](https://github.com/deepcomm/RNNViterbi). 





{% highlight python %}

from keras import backend as K

import tensorflow as tf

from keras.layers import LSTM, GRU, SimpleRNN

step_of_history = 200 # Length of input message sequence  

code_rate = 2 # Two coded bits per one message bit

num_rx_layer = 2 # Two layers of GRU 

num_hunit_rnn_rx = 50 # Each GRU has 50 hidden units



noisy_codeword = Input(shape=(step_of_history, code_rate)) 

x = noisy_codeword

for layer in range(num_rx_layer):

   x = Bidirectional(GRU(units=num_hunit_rnn_rx, 

​          activation='tanh',

​          return_sequences=True))(x)

   x = BatchNormalization()(x)

x = TimeDistributed(Dense(1, activation='sigmoid'))(x)

predictions = x

model = Model(inputs=noisy_codeword, outputs=predictions)

{% endhighlight %}



### Step 2. Supervised training -  optimizer, loss, and evaluation metrics 

So now we have an RNN based decoder model, which is nothing but a parametric function. We learn the parameters in a supervised matter, with examples of (noisy codeword **y**, message **b**), via backpropagation. The goal of training is to learn a set of hyperparameters so that the decoder model generates an estimate of **b** from **y** that is closest to the ground truth **b**.  Before we do the training, we have to choose [optimizer](https://keras.io/optimizers/) , [loss function](https://keras.io/losses/), and [evaluation metrics](https://keras.io/metrics/). Once we choose them, training a model is very simple. Summary of a model will show you how many parameters are in the decoder.

<!--- Training requires two step. 

* Step 1: choose [optimizer](https://keras.io/optimizers/) , [loss function](https://keras.io/losses/), and [evaluation metrics](https://keras.io/metrics/). Once we choose them, training a model is very simple! 
* Step 2: train the model with --->

{% highlight python %}

optimizer= keras.optimizers.adam(lr=learning_rate,clipnorm=1.)

#### custom evaluation metric: BER 

def errors(y_true, y_pred):

​    ErrorTensor = K.not_equal(K.round(y_true), K.round(y_pred))

​    return K.mean(tf.cast(ErrorTensor, tf.float32))



model.compile(optimizer=optimizer,loss='mean_squared_error', metrics=[errors])

print(model.summary())

{% endhighlight %}



### Step 3. Supervised training 

For training, we generate training examples, pairs of (noisy codeword, true message). To generate each example, we (i) generate a random bit sequence (of length *step_of_history*), (ii) generate convolutional coded bits using  [commpy](https://github.com/veeresht/CommPy), an open source, and (iii) add random noise of Signal-to-Noise Ratio *SNR*. Choosing the right parameters for *step_of_history* and *SNR* is critical in learning a reliable decoder. The variable k_test refers to how many bits in total will be generated. 



{% highlight python %}

#### Generate pairs of (noisy codwords, true message sequence)

noisy_codewords, true_messages, _ = generate_examples(k_test=k, step_of_history=200, SNR=0) 

{% endhighlight %}



The actual training is done with one line of code. 



{% highlight python %}

#### Train using the pre-defined optimizer and loss function 

model.fit(x=noisy_codewords, y=true_messages, batch_size=train_batch_size,

​                callbacks=[change_lr],

​                epochs=num_epoch, validation_split=0.1)

{% endhighlight %}



### Step 4. Test on various SNRs

We test the decoder on various SNRs, from 0 to 6dB. 



{% highlight python %}

TestSNRS = np.linspace(0, 6, 7, dtype = 'float32')

for idx in range(0,SNR_points):

​    TestSNR = TestSNRS[idx] 

​    noisy_codewords, true_messages, target = generate_examples(k_test=k_test,step_of_history=step_of_history,SNR=TestSNR) 

​    estimated_message_bits = np.round(model.predict(noisy_codewords, batch_size=test_batch_size))

​    ber = 1- sum(sum(estimated_message_bits == target))*\

​           1.0/(target.shape[0] * target.shape[1] *target.shape[2]) # target: true messages reshaped 

​    print(ber)





{% endhighlight %}



We provide a [MATLAB code](https://github.com/deepcomm/RNNViterbi/blob/master/viterbi_comparison.m) that implements Viterbi decoding for convolutional codes for readers who would like to run Viterbi on their own. For SNRs 0 to 6, the Bit Error Rate (BER) and the Block Error Rate (BLER) of the learnt decoder and Viterbi decoder are comparable. Note that we trained the decoder at SNR 0dB, but it has learned to do an optimal decoding for SNR 0 to 6dB. We also look at generalization across different code lengths; we see that for arbitrary lenght of messages, the error probability of neural decoder is indistinguishable from the one of Viterbi decoder. 

<center><img src="https://deepcomm.github.io/images/BER.png"></center>

<center><img src="https://deepcomm.github.io/images/BLER.png"></center>



### References 

[Communication Algorithms via Deep Learning](https://openreview.net/pdf?id=ryazCMbR-), Hyeji Kim, Yihan Jiang, Ranvir Rana, Sreeram Kannan, Sewoong Oh, Pramod Viswanath. ICLR 2018 





<div id="disqus_thread"></div>
<script>

/**
*  RECOMMENDED CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION BELOW TO INSERT DYNAMIC VALUES FROM YOUR PLATFORM OR CMS.
*  LEARN WHY DEFINING THESE VARIABLES IS IMPORTANT: https://disqus.com/admin/universalcode/#configuration-variables*/
/*
var disqus_config = function () {
this.page.url = PAGE_URL;  // Replace PAGE_URL with your page's canonical URL variable
this.page.identifier = PAGE_IDENTIFIER; // Replace PAGE_IDENTIFIER with your page's unique identifier variable
};
*/
(function() { // DON'T EDIT BELOW THIS LINE
var d = document, s = d.createElement('script');
s.src = 'https://hyejikim1-github-io.disqus.com/embed.js';
s.setAttribute('data-timestamp', +new Date());
(d.head || d.body).appendChild(s);
})();
</script>
<noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript">comments powered by Disqus.</a></noscript>


