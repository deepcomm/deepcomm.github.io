---

layout:     post
title:      Decoding convolutional codes 
date:       2020-02-01 12:31:19
summary:    Learning Viterbi Maximum Likelihood decoders for convolutional codes
categories: jekyll pixyll
comments:   true
---

Viterbi algorithm exactly computes the Maximum Likelihood (ML) estimate of the transmitted codeword, 
by efficiently running a dynamic programming. 
Although statistically this cannot be improved upon, 
we train a deep neural network to re-discover a Viterbi-like *algorithm* that matches the ML performance. 
Here the emphasis is on "algorithm"; we are only interested in an outcome of the machine learning process 
that can be readily applied as is to any block lengths, beyond what it was trained on. 
We do not consider a trained decoder that only works for a fixed block length as an algorithm.

The purpose of this exercise is two fold. 
First, our eventural goal is the discovery of new codes (encoder and decoder). 
Deomnstrating that deep learning can reproduce the optimal decoder for existing codes is a necessary intermediate step. 
Next, we want to impose additional constraints on the decoder, such as low latency. 
Deep learning provides such flexibilitym which Viterbi algorithm is not equipped with. 

A communication system has an encoder that maps messages (e.g., bit sequences) to codewords, typically of longer lengths, 
and a decoder that maps noisy codewords to the estimate of messages, as illustrated below. 





<center><img src="https://deepcomm.github.io/images/commsystem.png" width="750"/></center>

The design of channel codes is directly related to the reliability of communication systems; a practical value of reliable codes is enormous. The design of codes is also theoretically challenging and interesting; it has been a major area of study in information theory and coding theory for several decades since Shannon's 1948 seminal paper. 



As a first step towards revolutionizing channel coding via deep learning (e.g., learning a novel pair of encoder-decoder), we ask the very first natural question: **Can we learn an optimal decoder for a fixed encoder from data?**  To answer this question, we fix the encoder as one of the standard encoders, model the decoder as a neural network, and train the decoder in a supervised manner. 

<center><img src="https://deepcomm.github.io/images/learndec.png" width="750"/></center>

## Sequential code

When we fix the encoder, among many standard codes, we choose sequential codes such as *convolutional codes* and *turbo codes* for the following reasons:

* These codes are practical. They are used for mobile communications (e.g., 4G LTE) and satellite communications. 
  
* These codes achieve performance close to the fundamental limit.
  
* The recurrent nature of sequential encoding aligns very well with the Recurrent Neural Network (RNN) structure. 
  
* Well-known decoders exist for these codes. For convolutional codes, maximum likelihood decoder on AWGN channels is Viterbi decoder, which is a dynamic programming. For turbo codes, a belief propagation decoder on AWGN channels works extremely well. Hence, learning a decoder for sequential codes poses the challenge in *learning an algorithm.*



## Connection between sequential codes and RNNs

Below is an illustration of a *sequential* code that maps a message bit sequence **b** of length K to a codeword sequence **c**. The encoder first takes the first bit b<sub>1</sub>, update the state s<sub>1</sub>, and the generate coded bits **c<sub>1</sub>** based on the state s<sub>1</sub>. The encoder then takes the second bit b<sub>2</sub>, generate state s<sub>2</sub> based on (s<sub>1</sub>, b<sub>2</sub>), and then generate coded bits **c<sub>2</sub>**. These mappings occur recurrently until the last coded bits **c<sub>k</sub>** are generated. Each coded bits **c<sub>k</sub>** (k=1, ... K) is of length r when code rate is 1/r. For example, for code rate 1/2, **c<sub>1</sub>** is of length 2. 



<center><img src="https://deepcomm.github.io/images/seqcode.png"></center>







Recurrent Neural Network (RNN) is a neural architecture that's well suited for sequential mappings with memory. There is a hidden state h evolving through time. The hidden state keeps information on the current and all the past inputs. The hidden state is updated as a function (f) of previous hidden state and the input at each time. The output then is another function (g) of the hidden state at time k. In RNN, these f and g are parametric functions. Depending on what parametric functions you choose, the RNN can be a vanilla RNN, or LSTM, or GRU. See [here](http://dprogrammer.org/rnn-lstm-gru) for a detailed description. And once you choose the parametric function, we then learn a good parameter through training. So the RNN is a very natural fit to the sequential encoders. 

<center><img src="https://deepcomm.github.io/images/RNN.png"></center>



## Convolutional code and Viterbi decoding

Convolutional code and turbo codes are examples of sequential codes. Convolutional code is proposed by ... Turbo code is an extension of convolutional codes ... proposed by ... In this post, we focus on convolutional codes and learning the decoder for them; we will look into turbo codes in the next post. 

### Convolutional coding

An example for a rate 1/2 convolutional code is shown below. This code maps  b<sub>k</sub> to  (c<sub>k1</sub>, c<sub>k2</sub>), where the state is  (b<sub>k</sub>,  b<sub>k-1</sub>,  b<sub>k-2</sub>), and coded bits (c<sub>k1</sub>, c<sub>k2</sub>) are convolution (i.e., mod 2 sum) of the state bits.  

<center><img src="https://hyejikim1.github.io/images/convcode.png"></center>





### Viterbi decoding

Well-known decoders exist for these codes. For convolutional codes, maximum likelihood decoder on AWGN channels is Viterbi decoder, which is a dynamic programming. Viterbi decoder is propsoed by Andrew Viterbi in 19xx yy years after convolutional codes are proposed.  Hence, learning a decoder for convolutional codes poses the challenge in *learning an algorithm.*



will add1: Three-line summary of viterbi decoding - dynamic programming 

will add2: Reference to a good tutorial on Viterbi decoding 

Can we learn an optimal decoder for this convolutional code on AWGN channels? We will now walk you through with example codes. Full code can be accessed [here](https://github.com/deepcomm/RNNViterbi). 



## Learning an RNN decoder for convolutional codes

### Defining an RNN decoder

The first thing to do is to model the decoder as a neural network. We model the decoder as a Bi-directional RNN that has a forward pass and a backward pass. This is because we would like the decoder to estimate each bit based on the whole received sequence. In particular, we use GRU; empirically, we see GRU and LSTM have similar performance. The input to each 1st layer GRU cell is received coded bits, i.e., noise sequence **n<sub>k</sub>**  added to the k-th coded bits **c<sub>k</sub>**. The output of each 2nd layer GRU cell is the estimate of b<sub>k</sub>. 



<center><img src="https://hyejikim1.github.io/images/twolayerbiGRUDec.png"></center>

Here is an excerpt of python code that defines the decoder neural network. In this post, we introduce codes built on Keras library, which is arguably one of the easiest deep learning libraries, as a gentle introduction to deep learning programming for channel coding. 



For installation, do this and that. 



{% highlight python %}

from keras import backend as K

import tensorflow as tf

from keras.layers import LSTM, GRU, SimpleRNN



block_length = 100 # Length of input message sequence  

code_rate = 2 # Two coded bits per one message bit

num_rx_layer = 2

num_hunit_rnn_rx = 50

noisy_codeword = Input(shape=(step_of_history, code_rate)) # size is (100, 2) 

x = noisy_codeword

for layer in range(num_rx_layer):

   x = Bidirectional(GRU(units=num_hunit_rnn_rx, 

​          activation='tanh',

​          return_sequences=True))(x)

   x = BatchNormalization()(x)

x = TimeDistributed(Dense(1, activation='sigmoid'))(x)

Predictions = x

model = Model(inputs=noisy_codeword, outputs=predictions)

{% endhighlight %}



### Defining optimizer, loss, and evaluation metrics 

Will do 5: add more descriptions 

{% highlight python %}

optimizer= keras.optimizers.adam(lr=learning_rate,clipnorm=1.)



def errors(y_true, y_pred):

​    myOtherTensor = K.not_equal(K.round(y_true), K.round(y_pred))

​    return K.mean(tf.cast(myOtherTensor, tf.float32))



model.compile(optimizer=optimizer,loss='mean_squared_error', metrics=[errors])

print(model.summary())

{% endhighlight %}



### Training 

{% highlight python %}

model.fit(x=train_tx, y=X_train, batch_size=train_batch_size,

​          callbacks=[change_lr]

​          epochs=num_epoch, validation_split=0.1)  # starts training

{% endhighlight %}



### Results 



Will do 6: GRAPH 



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



<!---

There is a significant amount of subtle, yet precisely calibrated, styling to ensure
that your content is emphasized while still looking aesthetically pleasing.

All links are easy to [locate and discern](https://hyejikim1.github.io/), yet don't detract from the [harmony
of a paragraph](#). The _same_ goes for italics and __bold__ elements. Even the the strikeout
works if <del>for some reason you need to update your post</del>. For consistency's sake,
<ins>The same goes for insertions</ins>, of course.

### Code, with syntax highlighting



Here's an example of some ruby code with line anchors.

{% highlight ruby lineanchors %}
# The most awesome of classes
class Awesome < ActiveRecord::Base
  include EvenMoreAwesome

  validates_presence_of :something
  validates :email, email_format: true

  def initialize(email, name = nil)
    self.email = email
    self.name = name
    self.favorite_number = 12
    puts 'created awesomeness'
  end

  def email_format
    email =~ /\S+@\S+\.\S+/
  end
end
{% endhighlight %}

Here's some CSS:

{% highlight css %}
.foobar {
  /* Named colors rule */
  color: tomato;
}
{% endhighlight %}

Here's some JavaScript:

{% highlight js %}
var isPresent = require('is-present')

module.exports = function doStuff(things) {
  if (isPresent(things)) {
    doOtherStuff(things)
  }
}
{% endhighlight %}

Here's some HTML:

{% highlight html %}
<div class="m0 p0 bg-blue white">
  <h3 class="h1">Hello, world!</h3>
</div>
{% endhighlight %}

# Headings!

They're responsive, and well-proportioned (in `padding`, `line-height`, `margin`, and `font-size`).
They also heavily rely on the awesome utility, [BASSCSS](http://www.basscss.com/).

##### They draw the perfect amount of attention

This allows your content to have the proper informational and contextual hierarchy. Yay.

### There are lists, too

  * Apples
  * Oranges
  * Potatoes
  * Milk

  1. Mow the lawn
  2. Feed the dog
  3. Dance

### Images look great, too

![desk](https://cloud.githubusercontent.com/assets/1424573/3378137/abac6d7c-fbe6-11e3-8e09-55745b6a8176.png)

_![desk](https://cloud.githubusercontent.com/assets/1424573/3378137/abac6d7c-fbe6-11e3-8e09-55745b6a8176.png)_


### There are also pretty colors

Also the result of [BASSCSS](http://www.basscss.com/), you can <span class="bg-dark-gray white">highlight</span> certain components
of a <span class="red">post</span> <span class="mid-gray">with</span> <span class="green">CSS</span> <span class="orange">classes</span>.

I don't recommend using blue, though. It looks like a <span class="blue">link</span>.

### Footnotes!

Markdown footnotes are supported, and they look great! Simply put e.g. `[^1]` where you want the footnote to appear,[^1] and then add
the reference at the end of your markdown.

### Stylish blockquotes included

You can use the markdown quote syntax, `>` for simple quotes.

> Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse quis porta mauris.

However, you need to inject html if you'd like a citation footer. I will be working on a way to
hopefully sidestep this inconvenience.

<blockquote>
  <p>
    Perfection is achieved, not when there is nothing more to add, but when there is nothing left to take away.
  </p>
  <footer><cite title="Antoine de Saint-Exupéry">Antoine de Saint-Exupéry</cite></footer>
</blockquote>


### Tables

Tables represent tabular data and can be built using markdown syntax.  They are rendered responsively in Pixyll for a variety of screen widths.

Here's a simple example of a table:

| Quantity | Description |     Price |
|----------+-------------+----------:|
|        2 |      Orange |     $0.99 |
|        1 |   Pineapple |     $2.99 |
|        4 |      Banana |     $0.39 |
|==========|=============|===========|
|          |   **Total** | **$6.14** |

A table must have a body of one or more rows, but can optionally also have a header or footer.

The cells in a column, including the header row cell, can either be aligned:

- left,
- right or
- center.

Most inline text formatting is available in table cells, block-level formatting are not.

|----------------+----------------------+------------------------+----------------------------------|
| Default header | Left header          |     Center header      |                     Right header |
|----------------|:---------------------|:----------------------:|---------------------------------:|
| Default        | Left                 |        Center          |                            Right |
| *Italic*       | **Bold**             |   ***Bold italic***    |                      `monospace` |
| [link text](#) | ```code```           |     ~~Strikeout~~      |              <ins>Insertion<ins> |
| line<br/>break | "Smart quotes"       | <mark>highlight</mark> | <span class="green">green</span> |
| Footnote[^2]   | <sub>subscript</sub> | <sup>superscript</sup> |     <span class="red">red</span> |
|================+======================+========================+==================================+
| Footer row                                                                                        |
|----------------+----------------------+------------------------+----------------------------------|

### There's more being added all the time

Checkout the [Github repository](https://github.com/johnotander/pixyll) to request,
or add, features.

Happy writing.

---

[^1]: Important information that may distract from the main text can go in footnotes.

[^2]: Footnotes will work in tables since they're just links.



-->
