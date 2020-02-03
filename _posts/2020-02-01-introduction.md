---
layout:     post
title:       <font size="6"> Introduction and Mission Statement </font> 
date:       2020-02-01 01:21:29
summary:    Introduction and mission statement 
comments:   true
---


<font size="3">
--[*Pramod Viswanath*](https://pramodv.ece.illinois.edu)


Reliable digital communication, both wireline (ethernet, cable and DSL modems) and wireless (cellular, Wifi, satellite), is a primary workhorse of the modern information age. A critical aspect of reliable communication involves the design of codes that allow transmissions to be robustly (and computationally efficiently) decoded under noisy conditions. This is the discipline of communication theory; over the past century and especially the past 70 years (since the birth of information theory (Shannon, 1948)) much progress has been made in the design of near optimal codes. Landmark codes include convolutional codes, turbo codes, low density parity check (LDPC) codes and polar codes. The impact on humanity is enormous – every cellular phone designed uses one of these codes, which feature in global cellular standards ranging from the 2nd generation to the 5th generation respectively, and are text book material. 


The canonical setting is one of point-to-point reliable communication over the additive white Gaussian noise (AWGN) channel and performance of a code in this setting is its gold standard. The AWGN channel fits much of wireline and wireless communications although the front end of the receiver may have to be specifically designed before being processed by the decoder (example: intersymbol equalization in cable modems, beamforming and sphere decoding in multiple antenna wireless systems); again this is text book material (Tse & Viswanath, 2005). There are two long term goals in coding theory: (a) design of new, computationally efficient, codes that improve the state of the art (probability of correct reception) over the AWGN setting. Although the current codes already operate close to the (finite block length) information theoretic “Shannon limit”, there is room to improve the performance in some regimes (at "moderate" block lengths, high rates). Thre is also an emphasis  on robustness and adaptability to deviations from the AWGN settings (such as in urban, pedestrian, vehicular settings).  (b) design of new codes for multi-terminal (i.e., beyond point-to-point) settings – examples include the feedback channel, the relay channel and the interference channel.

Progress over these long term goals has generally been driven by individual human ingenuity and, befittingly, is sporadic. For instance, the time duration between convolutional codes (2nd generation cellular standards) to polar codes (5th generation cellular standards) is over 4 decades. Deep learning is fast emerging as capable of learning sophisticated algorithms from observed data (input, action, output) alone and has been remarkably successful in a large variety of human endeavors (ranging from language (Mikolov et al., 2013) to vision (Russakovsky et al., 2015) to playing Go (Silver et al., 2016)). Motivated by these successes, we envision that deep learning methods can play a crucial role in solving both the aforementioned goals of coding theory. The posts in this forum explore the research efforts to invent new communication algorithms via deep learning. The audience is two fold: on the one hand, machine leraning and deep learning researchers interested in "learning augmented algorithm design" and on the other hand, researches from information theory, wireless communication and coding theory. Given the two communities are fairly well separated (both in research and implementation techniques and in publication venues), this forum aims to bring the two together. We will do this by making the posts didactic in ways a traditional research summary would not. In particular, we aim to provide a detailed set of instructions to emulate the end-to-end learning process (data generation, optimization formulation, learning algorithms, normalizations, step sizes-- in short, all the details of the "dark arts" of deep learning methods).   
</font>

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
s.src = 'https://deepcomm-comments.disqus.com/embed.js';
s.setAttribute('data-timestamp', +new Date());
(d.head || d.body).appendChild(s);
})();
</script>
<noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript">comments powered by Disqus.</a></noscript>
