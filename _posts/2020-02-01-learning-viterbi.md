---

layout:     post
title:      Channel decoding via deep learning (1)
date:       2020-02-01 12:31:19
summary:    Learning Viterbi Maximum Likelihood decoders for convolutional codes
categories: jekyll pixyll
comments:   true
---



The first series of posts will cover applications of deep learning to channel coding, which is a basic building block of communication systems. An encoder maps messages (e.g., bit sequence) to codewords, typically of longer lengths. A decoder maps noisy codewords to the estimate of messages, as illustrated in a simplified figure below 

![desk](https://hyejikim1.github.io/images/commsystem.png)

The design of channel codes is directly related to the reliability of communication systems; practical value of better codes is enormous. The design of codes is also theoretically challenging and interesting; it has been a major area of study in information theory and coding theory for several decades since Shannon's 1948 seminal paper. 



As a first step towards revolutionizing channel coding via deep learning, we ask the very first natural question. **Can we learn optimal decoders solely from data?**



### Learning channel decoders



We fix the encoder as one of the standard encoders and learn the decoder for practical channels. When we fix the encoder, there’re many possible choices - and we choose sequential codes, such as convolutional codes and turbo codes. There are many reasons to it. First of all, these codes are practical. These codes are actually used for mobile communications as in 4G LTE, and satellite communications. Secondly, these codes achieve performance close to the fundamental limit, which is a very strong property. Lastly, the recurrent nature of sequential encoding process aligns very well with the recurrent neural network structure. Let me elaborate on this. 

![desk](https://deepcomm.github.io/images/learndec.png)



### Sequential code

We're going to show you an illusration of sequential code that maps a message sequence b to a codeword sequence c. We first take the first bit b<sub>1</sub>, and update the state s<sub>1</sub>, and the generate coded bits c<sub>1</sub> by looking at the state. Depending the rate of your code, c<sub>1</sub> can be of length 2 if it’s rate 1/2 or length 3 if it’s rate 1/3. And then you take the second bit b<sub>2</sub>, then you update your state s<sub>2</sub> based on s<sub>1</sub> and b<sub>2</sub>, and then geenrate coded bits c<sub>2</sub>. And you do this recurrently, until you map the last bit b<sub>K</sub> to the coded bit c<sub>K</sub>. 



<center><img src="https://hyejikim1.github.io/images/seqcode.png"></center>



### Convolutional code 

Convolutional code is an example of sequential codes. Here’s an example for a rate 1/2 convolutional code. Which maps bk to ck1 and ck2. The state is bk, bk-1, bk-2. Then the coded bits are convolution (or mod 2 sum) of the state bits. 



<center><img src="https://hyejikim1.github.io/images/convcode.png"></center>



### Recurrent Neural Network



Okay now let’s look at the Reccurent Neural Network architecture — (RNN in short) is a good neural architecture for sequential mappings with memory. 

The way it works is there is a hidden state h evolving through time. The hidden state keeps some information about the current and all the past inputs. The hidden state is updated as a function of previous hidden state and the input at the time. Then the output is another function of the hidden state at time i. 

In RNN, these f and g are some parametric functions. Depending on what parameteric functions you choose, the RNN can be a vanilla RNN, or LSTM, or GRU. And once you choose the parametric function, we then learn a good parameter through training.

So the RNN is a very natural fit to the sequential encoders. 

<center><img src="https://hyejikim1.github.io/images/RNN.png"></center>

### Viterbi decoder

Now when it comes to decoding, for these sequential codes, there are well known decoders under AWGN settings - such as Viterbi and BCJR decoders … 

### Modelling an RNN decoder 

The first thing to do is to model the decoder as a neural network. We model the decoder as a bi-directional RNN because the encoder is sequential. We model the decoder as a Bi-directional RNN (which has forward pass and baackward pass) because we’d like the decoder to look at the whole received sequence to estimate a certain bit. 



<center><img src="https://hyejikim1.github.io/images/twolayerbiGRUDec.png"></center>



{% highlight python %}

from keras import backend as K

import tensorflow as tf

from keras.layers import LSTM, GRU, SimpleRNN



block_length = 100 # Length of input message sequence  

code_rate = 2 # Two coded bits per one message bit

num_rx_layer = 2

num_hunit_rnn_rx = 50

noisy_codeword = Input(shape=(step_of_history, code_rate)) # size is (100, 2) - notation! 

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



GRAPH 



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
