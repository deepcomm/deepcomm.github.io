---
layout:     post
title:      Decoding convolutional codes
date:       2020-02-01 12:31:19
summary:    Learning Viterbi decoder.
categories: jekyll pixyll
---



Channel coding and decoding are basic building blocks of communication systems. An encoder maps a message to a codeword, where the codeword has some redundancy. A decoder maps noisy codeword to estimate of an message. 



### Channel encoding and decoding 

![desk](https://hyejikim1.github.io/images/commsystem.png)



We fix the encoder as one of the standard encoders and learn the decoder for practical channels. When we fix the encoder, there’re many possible choices - and we choose sequential codes, such as convolutional codes and turbo codes. There are many reasons to it.



First of all, these codes are practical. These codes are actually used for mobile communications as in 4G LTE, and satellite communications. Secondly, these codes achieve performance close to the fundamental limit, which is a very strong property. Lastly, the recurrent nature of sequential encoding process aligns very well with the recurrent neural network structure. Let me elaborate on this. 

### Sequential code

I’m gonna show you an illusration of sequential code that maps a message sequence b to a codeword sequence c. We first take the first bit b1, and update the state s1, and the generate coded bits c1 by looking at the state. Depending the rate of your code, c1 can be of length 2 if it’s rate 1/2 or length 3 if it’s rate 1/3. And then you take the second bit b2, then you update your state based on s1 and b2, and then geenrate coded bits c2. And you do this recurrently, until you map the last bit bK to the coded bit cK. 

![desk](https://hyejikim1.github.io/images/seqcode.png)

### Convolutional code 

Convolutional code is an example of sequential codes. Here’s an example for a rate 1/2 convolutional code. Which maps bk to ck1 and ck2. The state is bk, bk-1, bk-2. Then the coded bits are convolution (or mod 2 sum) of the state bits. 

![desk](https://hyejikim1.github.io/images/convcode.png)





### Recurrent Neural Network



Okay now let’s look at the Reccurent Neural Network architecture — (RNN in short) is a good neural architecture for sequential mappings with memory. 

The way it works is there is a hidden state h evolving through time. The hidden state keeps some information about the current and all the past inputs. The hidden state is updated as a function of previous hidden state and the input at the time. Then the output is another function of the hidden state at time i. 

In RNN, these f and g are some parametric functions. Depending on what parameteric functions you choose, the RNN can be a vanilla RNN, or LSTM, or GRU. And once you choose the parametric function, we then learn a good parameter through training.

So the RNN is a very natural fit to the sequential encoders. 



![desk](https://hyejikim1.github.io/images/RNN.png)

### Viterbi decoder

Now when it comes to decoding, for these sequential codes, there are well known decoders under AWGN settings - such as Viterbi and BCJR decoders … 

### RNN decoder 

The first thing to do is to model the decoder as a neural network. We model the decoder as a bi-directional RNN because the encoder is sequential. We model the decoder as a Bi-directional RNN (which has forward pass and baackward pass) because we’d like the decoder to look at the whole received sequence to estimate a certain bit. 



![desk](https://hyejikim1.github.io/images/twolayerbiGRUDec.png)



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
