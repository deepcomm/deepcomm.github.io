---
layout:     post
title:      Estimation of Rate-Distortion Function for Computing with Decoder Side Information
date:       2024-07-17, 21:00:00
summary:    An algorithm to estimate rate-distortion function for computing with side information 
categories: jekyll pixyll
comments:   true
visible:    true
author:     Heasung Kim, Hyeji Kim, and Gustavo de Veciana
---


In the ever-evolving landscape of data science, the quest for efficient data compression methods remains a critical challenge. Compression not only saves storage space but also enhances data transmission efficiency, crucial for everything from streaming video to cloud storage. But how do we know when a compression method is truly efficient? The answer lies in understanding the *rate-distortion function*, a notion of the trade-off between compression rate and the distortion of the compressed data.

---
### Rate-Distortion Function
The rate-distortion function is a fundamental concept in information theory, representing the minimum data rate required to transmit or store information such that the target distortion (or loss of quality) remains below a certain threshold. It serves as a yardstick for the efficiency of compression algorithms.

This concept can be extended to encompass scenarios where side information, correlated with the input source, is available at the decoder, or at both the encoder and decoder. This adaptation is widely recognized as the Wyner-Ziv rate-distortion function. Additionally, the notion of the rate-distortion is further broadened by considering communication systems where the goal is to compute a specific target function, rather than just to recover the original source. Such applications are commonly referred to as **Coding for Computing**.


In this post, we are interested in the system model depicted below, where switch (A) remains open and switch (B) is closed. This system model focuses on minimizing the distortion between a target output $$Z$$ and its estimated counterpart $$\hat{Z}$$ through the optimized encoder and decoder modules. The encoder receives an input source $$X$$ and compresses it into a codeword $$U$$. The decoder, utilizing this codeword along with side information $$Y$$, produces the estimated output $$\hat{Z}$$. Notably, this framework permits the target output $$Z$$ to differ from the input source $$X$$, allowing $$Z$$ to be any functional output $$g(X,Y)$$ tailored to specific system requirements.

![alt text](https://github.com/Heasung-Kim/rate-distortion-side-information/blob/main/imgs/system_model_coding_for_computing.png?raw=true)

In this case, the rate-distortion function is given as follows.

$$R_{\text{D,C}}(D)=\text{min}_{q_{U|X}(u|x),f(u,y) :\mathbb{E}[d(Z,\hat{Z})]\le{D}}I(X;U|Y)\quad \quad(1)$$

As mentioned, the rate-distortion function provides valuable insights, such as determining the number of bits required to achieve a certain level of distortion $$D$$. This function has various applications, including measuring the value of side information and serving as a theoretical benchmark for evaluating the performance of compression algorithms.


---
### How to compute Rate-Distortion Function?

However, computing this function for real-world data, which often involves complex, high-dimensional distributions, is no small feat. Also, we often do not have a true underlying source distribution (unknown source distribution), and conventional iterative approaches (Blahut-Arimoto type algorithms) face limitations, especially when applied to high-dimensional or continuous sources. 

To address these issues, we propose a neural network-based direct estimation method for the rate-distortion function for computing with side information, along with applicable methodologies.

---
### Proposed Algorithm

We begin by reformulating the optimization term in $$(1)$$ as follows by using the convexity property of the rate-distortion function. For a given $$s$$, we can build a Lagrangian form optimziation problem as follows.

$$\min_{q_{U|X}, f} \Big\{ \mathbb{E}_{X,Y,U}\Big[\log\frac{q_{U|X}(U|X)}{q_{U|Y}(U|Y)} \Big] - s\mathbb{E}[d(Z,\hat{Z})] \Big\}\quad \quad(2)$$ 

where 
$$q_{U|Y}(u|y) = \sum_{x\in \mathcal{X} } p_{X|Y}(x|y) q_{U|X}(u|x)$$ 
when $$X$$ is a discrete random variable and $$\hat{Z} = f(U,Y)$$.

By solving the Lagrangian form of the optimization problem, the resulting optimization variables (encoder and decoder) correspond to specific points on the true rate-distortion function. This means that if we solve equation $$(2)$$ for various values of the Lagrange multiplier $$s$$, we can obtain multiple points on the rate-distortion curve. By collecting these points, we can ultimately estimate the entire rate-distortion function, providing a comprehensive understanding of the trade-offs between data rate and distortion.

To facilitate estimation using a given dataset without the knowledge of the closed-form expression of the input source, we parameterize the optimization variables with neural networks as follows.

$$q_{U|X}(u|x) \approx q_{U|X}(u|x;\boldsymbol{\theta}_{\text{po}})$$

$$f(u,y)  \approx f(u,y;\boldsymbol{\theta}_{\text{dec}})$$

Then we minimize the objective $$(2)$$ by updating the set of parameters $$\boldsymbol{\theta}_{\text{po}}$$ and $$\boldsymbol{\theta}_{\text{dec}}$$.

### *Challenge*
However, note that the problem (2) involves $$q_{U|Y}(u|y)$$ which can be represented as follows.

$$q_{U|Y }(u|y) = \sum_{x\in \mathcal{X}} p_{X|Y}(x|y)  q_{U|X,Y}(u|x,y) \quad\quad(3)$$.

The efficient computation of $$q_{U|Y }(u|y)$$ is critical, as it needs to be executed for multiple instances to obtain the average of the log probability. However, this computation of presents a substantial challenge due to the unknown nature of the source distribution, computing the sum over $$\mathcal{X}$$ is non-trivial when domain $$\mathcal{X}$$ is a high-dimensional space and the data instances are limited.



### *Solution*
By using the following fact,

$$\argmin_{\hat{q}_{U|Y} } \mathbb{E}_{X,Y,U} \left[\log \frac{q_{U|X}(U|X; \boldsymbol{\theta}_{\text{po}})}{\hat{q}_{U|Y}(U|Y)}\right] = q_{U|Y;\boldsymbol{\theta}_{\text{po}} }$$

we also parameterize $$q_{U|Y}(U|Y)$$ by using a set of parameters $$\boldsymbol{\theta}_{\text{pr}}$$ as $$q_{U|Y}(U|Y;\boldsymbol{\theta}_{\text{pr}})$$ and solve an equivalent problem 

$$\min_{\boldsymbol{\theta}_{\text{po}}, \boldsymbol{\theta}_{\text{pr}}, \boldsymbol{\theta}_{\text{dec}}} \Big\{ \mathbb{E}_{X,Y,U}\Big[\log\frac{q_{U|X}(U|X; \boldsymbol{\theta}_{\text{po}} )}{q_{U|Y}(U|Y; \boldsymbol{\theta}_{\text{pr}})} \Big] - s\mathbb{E}[d(Z, f(U,Y,  \boldsymbol{\theta}_{\text{dec}} )] \Big\}\quad\quad(4)$$.

By incorporating all the main components of the optimization problem as optimization variables, we can perform gradient-based updates. In this process, the weights are updated in the direction that minimizes the objective function $$(4)$$.




---

### Verifying Algorithm's performance - 2WGN

How do we verify the performance of our algorithm? One approach is to use special cases where the rate-distortion function for computing with side information is known in a **closed-form**. By comparing our algorithm's results with these known rate-distortion function, we can assess its accuracy.

We adopt a scenario featuring a 2-component White Gaussian Noise (2-WGN$$(P, \rho)$$) source, where $$(X, Y)$$ forms pairs of i.i.d. jointly Gaussian random variables. Each pair in the sequence $$(X_1, Y_1), (X_2, Y_2), \ldots, (X_n, Y_n)$$ is correlated by $$Y=X+W$$ where the distributions of $$X$$ and $$W$$ have zero mean, $$\mathbb{E}[X] = \mathbb{E}[W] = 0$$, and variance $$\mathbb{E}[X^2] = P$$ and $$\mathbb{E}[W^2] = N$$. With a squared error distortion measure $$d$$, the rate-distortion function $$R_{\text{D}}$$ is given as follows.

$$R_{\text{D,C}}(D) = \max\Big\{ \frac{1}{2}\log \Big( \frac{PN}{(P+N)D} \Big), 0\Big\}.$$


Our objective in this section is to apply our algorithm to estimate rate-distortion points and assess its accuracy in obtaining these points on the true rate-distortion curve.

For the parameterization of the distributions and the decoder, we used 2-layer Multi-Layer Perceptrons (MLPs). The specifics of the simulation setup, including hyperparameters and training details, can be found in our original paper.

#### Results


![alt text](https://github.com/Heasung-Kim/rate-distortion-side-information/blob/main/imgs/rd_plot_2wgn.png?raw=true)

In Fig. 3, we set $$P = 1, n = 100$$, and provide simulation results for various $$\rho$$ values in $$\{0.2, 0.4, 0.6, 0.8\}$$. Each subplot displays the $$R_{D,C}$$ curves, alongside four rate-distortion points estimated by our algorithm for different slopes $$s$$. We also plot $$R_{D,C}$$ curves with $$ρ=0$$. $$y$$-axis has natural units (Nats) and xaxis represents mean squared error distortion. The dashed lines associated with $$\hat{R}_{D,C}(D)$$ corresponds to the learning trajectory, i.e., the achieved (distortion, rate) points during the training process. For each of the subplots in the above figure,

Our algorithm consistently estimates the points on $$R_{D,C}$$ within a small tolerance! 


---
### Practical Applications - CSI Compression
Our algorithm can be applied to practical scenarios where estimating the rate-distortion function is crucial. One such application is the Channel State Information (CSI) compression problem in Frequency Division Duplex (FDD) communications, where a User Equipment (UE) communicates with a Base Station (BS). This problem is increasingly relevant in wireless research, as efficient CSI compression can significantly enhance communication system performance.

![alt text](https://github.com/Heasung-Kim/rate-distortion-side-information/blob/main/imgs/DL_CSI_UL_CSI.png?raw=true)


In more detail, our main objective is to compress the DL CSI, $$X$$, at the User Equipment (UE) side (DL CSI is depicted on the left hand side of the above image). The UE then transmits this compressed information, or codeword $$U$$, to the Base Station (BS). The aim is to minimize the Normalized Mean Squared Error (NMSE), defined as $$\mathbb{E}[{\Vert X-\hat{X} \Vert_{2}^{2}}/{\Vert X \Vert_{2}^{2} }]$$, where $$\hat{X}$$ is the decoder output and $$\Vert\cdot\Vert_{2}$$ is elementwise square norm.

To enhance compression efficiency, uplink (UL) CSI can be utilized as side information $Y$ (See right hand side of the above image). This approach leverages the fact that UL CSI is typically available at the BS side through pilot transmissions from the UE to the BS and is correlated with downlink (DL) CSI due to frequency-invariant characteristics.

We will apply our algorithm to estimate the rate-distortion function with decoder side information in this setting.

#### Results

In the figure below, four distinct curves are presented: the estimated rate-distortion curve with side information, $$\hat{R}_{\text{D,C}}$$, the one without side information, $$\hat{R}_{\text{C}}$$, the rate-distortion curve derived from the constructive compression algorithm with side information (compression with SI), and the curve without side information (compression).
The four points plotted on $$\hat{R}_{\text{D,C}}$$ and $$\hat{R}_{\text{C}}$$ denote distinct estimated rate-distortion points, with their positions corresponding to specific $$s$$ values: -100, -10, -1, and -0.1, arranged from left to right. $$\hat{R}_{\text{C}}$$, the estimated $${R}_{\text{C}}$$, is obtained by ignoring the side information. By adjusting $$s$$ values, we explore distortion levels from -8dB to approximately -23dB, connecting these points linearly to serve as an upper bound for the estimated rate-distortion curves. We also adopt VQ-VAE-based compression algorithm for purpose of comparison. The rate-distortion curves from the constructive algorithm with and without side information (light blue/yellow) are generated by varying the compression bit rates as $$l_{\text{cl}}\in \{16, 32, 64, 128\}$$ and connecting the points.


![alt text](https://github.com/Heasung-Kim/rate-distortion-side-information/blob/main/imgs/rd_plot_csi.png?raw=true)

As expected, introducing UL CSI (side information) for DL CSI compression is beneficial as $$\hat{R}_{\text{D,C}} < \hat{R}_{\text{C}}$$, especially at lower CSI feedback rates. For instance, in regions where the rate is below 10 Nats/sample, a gain of over 1 dB is expected from UL side information. This advantage diminishes with increased feedback resources; for example, at 150 Nats/Sample, the gain is observed to be near zero.


The neural compression algorithm incorporating side information, achieved a rate-distortion curve that establishes an upper bound of $$\hat{R}_{\text{D,C}}$$, and the discrepancy between $$\hat{R}_{\text{D,C}}$$ and the constructive CSI compression algorithm's performance signals room for improvement. For example, in the case of around 80 Nats/sample, we may anticipate a potential improvement about 1dB. Notably, this gap is less pronounced in scenarios with lower rates, allowing one to have a conjecture that the actual performance of the CSI compression algorithms is closer to $$\hat{R}_{\text{D,C}}$$. 


---
### Conclusion
We propose a new algorithm for estimating the generalized rate-distortion function, with a specific emphasis on the rate-distortion function for computing with side information.  This approach can offer estimated rates for given distortion levels and also enables the formulation of reliable conjectures about the benefits of side information at varying compression rates. Such a methodology is anticipated to be valuable in practical system design, allowing system designers to effectively measure the potential gains from side information against its processing costs through informed estimations.



For more details, please refer to the original paper [Estimation of Rate-Distortion Function for
Computing with Decoder Side Information.](https://openreview.net/forum?id=xDa9Dxoww0)

## References



A. El Gamal and Y.-H. Kim,  Network information theory.  Cambridge University Press, 2011.

H. Yamamoto, “Wyner-ziv theory for a general function of the correlated sources,” IEEE Transactions on Information Theory, vol. 28, no. 5, pp. 803–807, 1982.

H Kim, H Kim, G De Veciana, "Estimation of Rate-Distortion Function for Computing with Decoder Side Information", First Learn to Compress Workshop at ISIT 2024


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
