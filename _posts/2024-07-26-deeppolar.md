---
layout:     post
title:      Learning Non-Linear Polar codes via Deep Learning
date:       2024-07-26, 10:30:00
summary:    Generalizing polar encoding and decoding structures using neural networks leads to gains in BER
categories: jekyll pixyll
comments:   true
visible:    false
author:    Ashwin Hebbar, Hyeji Kim, and Pramod Viswanath
---

Introduction to the problem

## Setup : Channel coding
We consider the problem of reliably communicating a binary message over a noisy channel. The effect of noise can be mitigated by adding redundancy to the message. One simple method to achieve this is through repetition coding, where the same bit is sent multiple times; and we can reliably decode the bit through a majority vote. This process is known as channel coding, which involves an encoder that converts messages into higher-dimensional codewords, and a decoder that retrieves the original message from the noisy codewords, as depicted in the figure below. Over the years, numerous codes have been invented, including convolutional codes, turbo codes, LDPC codes, and more recently, polar codes. The impact of these codes has been tremendous; each of these codes have been part of global communication standards, and have powered the information age. At large blocklengths, these schemes operate close to information-theoretic limits. However, there is still room for improvement in the short-to-medium blocklength regime. The invention of codes has been sporadic, and primarily driven by human ingenuity. Recently, deep learning has achieved unparalleled success in a wide range of domains. We explore ways to automate the process of inventing codes using deep learning.
 

<center><img src="https://deepcomm.github.io/images/commsystem.png" width="750"/></center>

### Deep-Learning-based Channel codes

The search for good codes can be automated by parameterizing and learning both the encoder and decoder using neural networks. However, constructing effective non-linear codes using this approach is highly challenging: in fact, naively
using off-the-shelf neural architectures often
results in performance worse than even repetition codes.

Nevertheless, several recent works have introduced neural codes that match or outperform classical schemes. A common theme is the incorporation of structured redundancy, by using principled coding-theoretic encoding and decoding structures. For example, Turbo Autoencoder (Jiang et al, 2019) uses sequential encoding and decoding along with interleaving
of input bits, inspired by Turbo codes. KO codes (Makkuva et al., 2021) generalizes Reed-Muller encoding and Dumer decoding by replacing
selected components in the Plotkin tree with neural
networks. Product Autoencoder (Jamali et al., 2021b) generalizes
two-dimensional product codes to scale neural codes to
larger block lengths. In this post, we introduce DeepPolar, which generalizes the coding structures of large-kernel Polar codes by using
non-linear kernels parameterized by NNs.

### Polar codes
Polar codes, devised by Erdal Arikan in 2009, marked a significant breakthrough as the first class of codes with a deterministic construction proven to achieve channel capacity, via successive cancellation decoding. Further, in conjunction with higher complexity decoders, these codes also demonstrate good finite-length performance while maintaining relatively low encoding and decoding complexity. The impact of Polar codes is evident from their integration into 5G standards within just a decade of their proposal—a remarkably swift timeline (typically, codes take several decades to be adopted into cellular standards).

The basic building block of Polar codes is a binary matrix \( G_2 = \begin{pmatrix} 1 & 0 \\ 1 & 1 \end{pmatrix} \), known as the polarization kernel. To construct the encoding matrix for a block length of \(2^m\), we apply the Kronecker product to \( G_2 \) \(m\) times.
An alternate, efficient way to view Polar encoding is via the Plotkin transform : \( \{0,1\}^d \times \{0,1\}^d \to \{0,1\}^{2d} \), which transforms \( (u_0, u_1) \) to \( (u_0 \oplus u_1, u_1) \). This Plotkin transform is applied recursively on a tree to obtain the full Polar transform. This recursive application leads to a fascinating phenomenon called "channel polarization" - where each input bit encounters either a noiseless or highly noisy channel under successive cancellation decoding. This selective reliability allows us to transmit message bits through reliable positions only, while "freezing" less reliable positions with a known value, typically zero.

Polar codes can be decoded efficiently using the Successive Cancellation (SC) decoder. The basic principle behind the SC algorithm is to sequentially decode each message bit \(u_i\) according to the conditional likelihood given the corrupted codeword \(y\) and previously decoded bits \(\hat{\bu}^{(i-1)}\). The LLR for the \(i^{\text{th}}\) bit can be computed as
\begin{equation}
    L_i = \log \left( \frac{\mathbb{P}(u_i = 0 \mid y, \hat{\bu}^{(i-1)})}{\mathbb{P}(u_i = 1 \mid y, \hat{\bu}^{(i-1)})} \right).
\end{equation}

A detailed exposition of Polar coding and decoding can be found in [these notes](http://pfister.ee.duke.edu/courses/ecen655/polar.pdf).

<center><img src="https://deepcomm.github.io/images/deeppolar/deeppolar_transform.png" width="750"/></center>

### DeepPolar codes 
DeepPolar codes generalizes the encoding and decoding structures of Polar codes.
1) **Kernel Expansion:** The conventional \(2 \times 2\) Polar kernel is expanded into a larger \(\ell \times \ell\) kernel.
2) **Learning the kernel:** We parameterize each kernel by a learnable function \(g_\phi\), represented by a small Multilayer Perceptron (MLP). Likewise, we augment the SC decoder with learnable functions \(f_\theta\). The encoder-decoder pair is trained jointly to minimize the bit error rate (BER), using the binary cross entropy between transmitted and estimated messages as a surrogate loss function.


The expanded function space afforded by the non-linearity and the increased kernel size allows us to discover more reliable codes within the neural Plotkin code family.


Our experiments show that setting kernel size \(\ell = \sqrt{n}\) yields the best performance. This finding is consistent with the principles of the bias-variance tradeoff; while larger kernel provide expanded function spaces, it is harder to generalize with limited training examples.

### Results
TODO.

<center><img src="https://deepcomm.github.io/images/deeppolar/ber_plot.png" width="750"/></center>

To interpret the encoder, we examine the distribution of pairwise distances between codewords. Gaussian codebooks achieve capacity and are optimal asymptotically (Shannon, 1948). Remarkably,
the distribution of DeepPolar codewords closely resembles that of the Gaussian codebook. 

<center><img src="https://deepcomm.github.io/images/deeppolar/gaussian.png" width="750"/></center>

For further details, we encourage you to review our paper.

## References 

[DeepPolar: Inventing Nonlinear Large-Kernel Polar Codes via Deep Learning](https://arxiv.org/abs/2402.08864), Ashwin Hebbar, Sravan Kumar Ankireddy, Hyeji Kim, Sewoong Oh, Pramod Viswanath. ICML 2024





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
