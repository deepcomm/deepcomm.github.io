---
layout:     post
title:      Neural Cover Selection for Image Steganography
date:       2023-08-25, 21:00:00
summary:    Leveraging pretrained generative models to identify suitable cover images for data embedding
categories: jekyll pixyll
comments:   true
visible:    true
author:     Karl Chahine, Hyeji Kim
---


# Neural Cover Selection for Image Steganography
This repository contains the code for our paper [Neural Cover Selection for image Steganography](https://arxiv.org/abs/2410.18216) by Karl Chahine and Hyeji Kim (NeurIPS 2024). 

# Framework summary
Image steganography embeds secret bit strings within typical cover images, making them imperceptible to the naked eye yet retrievable through specific decoding techniques. The encoder takes as input a cover image ***x*** and a secret message ***m***, outputting a steganographic image ***s*** that appears visually similar to the original ***x***. The decoder then estimates the message ***m̂*** from ***s***. The setup is shown below, where _H_ and _W_ denote the image dimensions and the payload _B_ denotes the number of encoded bits per pixel (bpp).

<p style="margin-top: 30px;">
    <img src="steg_setup.png" alt="Model performance" width="600"/>
</p>

The effectiveness of steganography is significantly influenced by the choice of the cover image x, a process known as cover selection. Different images have varying capacities to conceal data without detectable alterations, making cover selection a critical factor in maintaining the reliability of the steganographic process.

Traditional methods for selecting cover images have three key limitations: (i) They rely on heuristic image metrics that lack a clear connection to steganographic effectiveness, often leading to suboptimal message hiding. (ii) These methods ignore the influence of the encoder-decoder pair on the cover image choice, focusing solely on image quality metrics. (iii) They are restricted to selecting from a fixed set of images, rather than generating one tailored to the steganographic task, limiting their ability to find the most suitable cover.

In this work, we introduce a novel, optimization-driven framework that combines pretrained generative models with steganographic encoder-decoder pairs. Our method guides the image generation process by incorporating a message recovery loss, thereby producing cover images that are optimally tailored for specific secret messages. We investigate the workings of the neural encoder and find it hides messages within low variance pixels, akin to the water-filling algorithm in parallel Gaussian channels. Interestingly, we observe that our cover selection framework increases these low variance spots, thus improving message concealment.

The DDIM cover-selection framework is illustrated below: 

<p style="margin-top: 30px;">
    <img src="DDIM_setup.png" alt="Model performance" width="600"/>
</p>

The initial cover image $\textbf{x}_0$ (where the subscript denotes the diffusion step) goes through the forward diffusion process to get the latent $\textbf{x}_T$ after _T_ steps. We optimize $\textbf{x}_T$ to minimize the loss ||***m*** - ***m̂***||. Specifically, $\textbf{x}_T$ goes through the backward diffusion process generating cover images that minimize the loss. We evaluate the gradients of the loss with respect to $\textbf{x}_T$ using backpropagation and use standard gradient based optimizers to get the optimal $\textbf{x}^*_T$ after some optimization steps. We use a pretrained DDIM, and a pretrained LISO, the state-of-the-art steganographic encoder and decoder from Chen et al. [2022]. The weights of the DDIM and the steganographic encoder-decoder are fixed throughout $\textbf{x}_T$'s optimization process.



For a deeper understanding and further details, we encourage you to review our research paper.


---

## References

[1] [DeepIC+: Learning Codes for Interference Channels](https://ieeexplore.ieee.org/document/10215318), Karl Chahine, Yihan Jiang, Joonyoung Cho, Hyeji Kim. IEEE Transactions on Wireless Communications, 2023.

[2] [Turbo Autoencoder: Deep learning based channel codes for point-to-point communication channels](https://arxiv.org/abs/1911.03038), Yihan Jiang, Hyeji Kim, Himanshu Asnani, Sreeram Kannan, Sewoong Oh, Pramod Viswanath. NeurIPS, 2019.

[3] [Similarity of neural network representations revisited](http://proceedings.mlr.press/v97/kornblith19a.html), Simon Kornblith, Mohammad Norouzi, Honglak Lee, Geoffrey Hinton. ICML, 2019.

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
