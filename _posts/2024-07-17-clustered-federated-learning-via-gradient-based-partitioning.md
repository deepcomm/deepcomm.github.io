---
layout:     post
title:      Learning context-dependent autoencoders for CSI feedback
date:       2024-07-17, 12:00:00
summary:    A clustered federated learning algorithm to cluster the clients and learn multiple models
categories: jekyll pixyll
comments:   true
visible:    true
author:     Heasung Kim, Hyeji Kim, and Gustavo de Veciana
---

### A Motivating Question: How do we cluster different datasets into different groups?

When developing data-driven machine learning models for a system, we often encounter scenarios where the data comes from not just one, but multiple heterogeneous distributions. In such cases, if the model's capacity is limited, it is beneficial to manage multiple models tailored to these distinct data distributions rather than relying on a single model for the entire system. Consider the following example.

####  Example: CSI compression models for heterogeneous datasets
<img src="https://github.com/Heasung-Kim/clustered-federated-learning-via-gradient-based-partitioning/blob/main/imgs/wireless_imgs/system_model.png?raw=true" alt="alt text"/>

Let's consider developing data-driven CSI feedback autoencoder models for the environment described above, where multiple data distributions (channel models for commercial, residential areas and parks...) arise from diverse environmental characteristics.
In designing a CSI feedback autoencoder, which consists of an encoder and a decoder, the encoder is deployed at the User Equipment (UE) and the decoder at the Base Station (BS). The encoder compresses the Channel State Information (CSI) and transmits the compressed data to the BS, where the decoder reconstructs the original CSI.

 An important question arises: *Which data distributions (areas, in this example) should be grouped together, allowing some to cooperate in learning while keeping others separate for optimal performance?*
One might want to group the commercial and residential areas together due to their high signal scattering characteristics. In contrast, channels in large parks are more likely to have a line-of-sight path - so the parks might have been grouped together.

#### In realistic system, clustering is highly challenging.
In practice, clustering the **clients** (i.e., the instance having dataset/distribution and participating in learning, in the example, area-BS pair) is challenging because we often lack detailed knowledge about the characteristics of each data distribution. In this simplified example, distributions are labeled as commercial, residential, or park. However, in reality, such (1) labels are not readily available. 
Furthermore, there may be scenarios where BSs are operated by different operators who are (2) unwilling to share their datasets but still seek to develop a robust model through collaboration. The lack of access to raw datasets further complicates the problem. is challenging because we often lack detailed knowledge about the characteristics of each data distribution. 


### Solution: Clustered Federated Learning
The Clustered Federated Learning (CFL) framework effectively addresses these challenges by clustering clients to share models without the need for dataset sharing. This approach allows for the creation of models tailored to each cluster. CFL aligns with the principles of Federated Learning (FL), where a central unit, connected to all clients, manages the model(s), transmits them to the clients, and collects the corresponding gradients. In this setup, only model weights and gradients are shared among clients and the central unit. By promoting cooperation among clients with similar local distributions, CFL results in more accurate, customized models without the necessity of aggregating data from all clients.

In this post, we introduce a novel and robust clustered federated learning algorithm, [*CFL-GP*](https://proceedings.mlr.press/v235/kim24p.html).

### CFL-GP (Clustered Federated Learning via Gradient-based Partitioning)


#### Core Idea of CFL-GP

The core concept of CFL-GP (Clustered Federated Learning via Gradient-based Partitioning) is that if clients have the same or similar data distributions, their gradients (i.e., the gradients from minibatch training with respect to the parameterized model) should also be similar. Therefore, the goal is to cluster the clients based on the similarity of their gradients.

#### Challenges of Gradient-Similarity Based Clustering

However, gradients are typically (1) *high-dimensional*, especially when using neural networks, and can be (2) *noisy* due to the stochastic nature of minibatch sampling.

#### Our Method
To overcome these challenges, we accumulate gradient information over multiple training steps and apply spectral clustering. When the stochastic gradient noise is modeled as a Gaussian distribution, this accumulation process gradually reduces the noise, leading to more accurate gradient-based clustering over time!



#### Example steps

<img src="https://github.com/Heasung-Kim/clustered-federated-learning-via-gradient-based-partitioning/blob/main/imgs/poster/cfl-gp-examplestep-1.JPG?raw=true" alt="alt text"/>

1.  **Broadcast Model:** The Central Unit (CU) manages multiple models, and picks one of them in a round-robbin manner and broadcasts the model $$\theta^{(t)}$$ to all clients.
2.  **Gradient Transmission:** The clients compute the corresponding gradients $$g(\theta^{(t)})$$ and send them back to the CU.
3.  **Gradient-Based Clustering:** The CU performs clustering based on the received gradient information. We use spectral clustering for this step, although the specific details are omitted here for simplicity.
4.  **Cluster driven model update:** Based on the clustering results, the clients in each group cooperate to train a model.

Relying solely on single, instantaneous gradient information can be noisy and may lead to incorrect clustering, as shown in the provided example. However, by accumulating gradient information over time, we can achieve more accurate clustering results - see the figure below.

<img src="https://github.com/Heasung-Kim/clustered-federated-learning-via-gradient-based-partitioning/blob/main/imgs/poster/cfl-gp-examplestep-2.JPG?raw=true" alt="alt text"/>

CFL-GP repeats this process at regular intervals during the federated learning process. By accumulating gradient information, the algorithm progressively refines the clustering, leading to better results!


### Experiment 

We consider five different data distributions (channel distributions) using the 3GPP channel model dataset generator, Quadriga ("quadriga-channel-model.de"), corresponding to Distributions 1 to 5.

 Configuration |   Distribution 1 ($$\mathcal{D}_{1}$$)    |   Distribution 2 ($$\mathcal{D}_{2}$$)   |    Distribution 3 ($$\mathcal{D}_{3}$$)   |    Distribution 4  ($$\mathcal{D}_{4}$$)  |    Distribution 5 ($$\mathcal{D}_{5}$$)     
---------------|:-------------------:|:-------------------:|:--------------------:|:--------------------:|:---------------------:|     
Channel model| 3GPP_38.901_UMi_LOS | 3GPP_38.901_UMi_LOS | 3GPP_38.901_UMi_NLOS | 3GPP_38.901_UMi_NLOS | 3GPP_38.901_UMi_NLOS  
Number of dominant reflection clusters|          5          |          5          |          35          |          40          |          10           


All channel distributions are generated at a center frequency of 2.53 GHz. The devices utilize a 3GPP-3D antenna model with dimensions of (1, 32). The number of subcarriers is 256, with a subcarrier spacing of 15000 Hz. Each instance in the antenna-frequency domain undergoes a 2-dimensional Inverse Fast Fourier Transform (IFFT) and is subsequently cropped in the high delay domain, resulting in each data instance being represented by a 32x32 complex matrix. A total of 30,000 data instances are extracted from each distribution, amounting to 150,000 instances in total.

#### Goal
The goal is to cluster $$N=40$$ clients (area-BS pairs), each with one of the channel distributions (Distributions 1 to 5), and train $$K=5$$ models. In summary, the central unit aims to cluster the clients with similar channel distributions without accessing the raw datasets.


#### Results
<img src="https://github.com/Heasung-Kim/clustered-federated-learning-via-gradient-based-partitioning/blob/main/imgs/wireless_imgs/fig_15_3.png?raw=true" alt="alt text"/>

The figure above shows the compression performance, normalized MSE in dB scale, and Adjusted Rand Index (ARI) across communication rounds. As baselines, we use [IFCA](https://proceedings.neurips.cc/paper_files/paper/2020/hash/e32cc80bf07915058ce90722ee17bb71-Abstract.html), [CFL:MADMO](https://ieeexplore.ieee.org/abstract/document/9174890), and [FedAvg](https://proceedings.mlr.press/v54/mcmahan17a?ref=https://githubhelp.com) for performance comparison. Our proposed algorithm achieves the highest clustering performance ARI, reaching the optimal ARI of 1.0. Corresponding to the high ARI, our method also demonstrates superior compression performance (lower MSE indicates better performance).

<img src="https://github.com/Heasung-Kim/clustered-federated-learning-via-gradient-based-partitioning/blob/main/imgs/wireless_imgs/fig_16.png?raw=true" alt="alt text"/>

We also plot the client features (the accumulated gradient information, randomly projected into 3D space), at $$t=1,160,320$$ and $$480$$. Each marker represents a client feature and is colored based on its underlying distribution. In the initial step, single gradient information may not be sufficient for clustering. However, as more gradient information is accumulated, clear clusters emerge, as seen at $$t=480$$ case.



### Conclusion
Our proposed method, CFL-GP, effectively clusters clients based on gradient similarity by accumulating gradient information. This accumulation helps denoise the gradient noise, ultimately leading to more accurate clustering results. In the example, better clustering enhances task performance, such as compression efficiency, by allowing the model to allocate the limited latent space (codeword space) more effectively. It ensures that similar data distributions share the same codeword space while different distributions utilize separate codeword spaces, optimizing overall system performance.






For more details of the algorithm, please refere to our paper [Clustered Federated Learning via Gradient-based Partitioning.](https://proceedings.mlr.press/v235/kim24p.html)


## References

Kim, Heasung, Hyeji Kim, and Gustavo De Veciana. "Clustered Federated Learning via Gradient-based Partitioning." Forty-first International Conference on Machine Learning (ICML), 2024

Ghosh, Avishek, et al. "An efficient framework for clustered federated learning." _Advances in Neural Information Processing Systems_ 33 (2020): 19586-19597.

Sattler, Felix, Klaus-Robert Müller, and Wojciech Samek. "Clustered federated learning: Model-agnostic distributed multitask optimization under privacy constraints." _IEEE transactions on neural networks and learning systems_ 32.8 (2020): 3710-3722.

McMahan, Brendan, et al. "Communication-efficient learning of deep networks from decentralized data." _Artificial intelligence and statistics_. PMLR, 2017.



<div id="disqus_thread"></div>
<script>

/**

* RECOMMENDED CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION BELOW TO INSERT DYNAMIC VALUES FROM YOUR PLATFORM OR CMS.
* LEARN WHY DEFINING THESE VARIABLES IS IMPORTANT: https://disqus.com/admin/universalcode/#configuration-variables*/
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
