---
layout:     post
title:      Turbo Decoding via Model-based ML
date:       2023-04-013 14:17:19
summary:    Boosting classical decoders by adding learnable parameters
categories: jekyll pixyll
comments:   true
visible:    false
author:    
---


The unparalleled success of deep learning in various domains has revolutionized the way we approach problem-solving in areas such as computer vision and natural language processing. More recently, there has been significant progress in adapting these techniques to perform channel decoding, achieveing impressive gains in performance and robustness.
However, these gains come at a cost, with deep learning models often requiring significant computational resources and vast amounts of training data. This limitation becomes particularly challenging for practical deployment of these systems on hardware-limited devices such as mobile phones and IoT systems, which have strict memory, computation, and latency constraints.

In recent years, many researchers have developed hybrid techniques that combine the advantages of both classical model-based methods, and deep learning algorithms. This paradigm, model-based machine learning, is a scalable way to use domain knowledge to obtain gains over traditional methods without a significant complexity increase. This blog post illustrates an application of model-based ML for improving the performance of Turbo decoding.

Consider communicating a message over a noisy channel.  This communication system has an encoder that maps messages (e.g., bit sequences) to codewords, typically of longer lengths, and a decoder that maps noisy codewords to the estimate of messages. This is illustrated below. 

<center><img src="https://deepcomm.github.io/images/commsystem.png" width="750"/></center>

<!-- Among the variety of existing codes, we start with a simple family of codes 
known as covolutional codes. Several properties make them ideal for our experiement. 

* These codes are practical (used in satellite communication) and form the building block of Turbo codes which are used for cellular communications. 
  
* With sufficient memory, the codes achieve performance close to the fundamental limit.
  
* The recurrent nature of sequential encoding aligns very well with a class of deep learning architectures known as Recurrent Neural Networks (RNN). 
  
* Well-known decoders exist for these codes. For convolutional codes, maximum likelihood decoder on AWGN channels is the Viterbi decoder, which is an instance of dynamic programming.  -->
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

### Turbo codes

Convolutional codes were introduced in 1955 by [Peter Elias](https://en.wikipedia.org/wiki/Peter_Elias). 
It uses  short memory and connvolution operators to sequentially create coded bits. 
An example for a rate 1/2 convolutional code is shown below. This code maps  b<sub>k</sub> to  (c<sub>k1</sub>, c<sub>k2</sub>), where the state is  (b<sub>k</sub>,  b<sub>k-1</sub>,  b<sub>k-2</sub>), and coded bits (c<sub>k1</sub>, c<sub>k2</sub>) are convolution (i.e., mod 2 sum) of the state bits.  

<center><img src="https://hyejikim1.github.io/images/convcode.png"></center>

<Introduce Turbo codes>





### Turbo decoding

\< Section about Turbo decoding \>

\< Explain BCJR \>

\< In practice, we use max-log-MAP. Can we do better? \>



### References 

[TinyTurbo: Efficient Turbo Decoders on Edge](https://arxiv.org/abs/2209.15614), Ashwin Hebbar, Rajesh Mishra, Sravan Kumar Ankireddy, Ashok Makkuva, Hyeji Kim, Pramod Viswanath. ISIT 2022





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


