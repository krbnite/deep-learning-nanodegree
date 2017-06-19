



In the neural network lingo, an AE is a triplet like (encoder, decoder, loss function).  
The encoder and decoder are both neural networks.  

The encoder is a projection operator (in QM lingo), or compression scheme (in DSP lingo).  
Its job is to project N-dimensional input onto a M-dimensional space, where M < N. 

The decoder is just the encoder in reverse.  

Example: One can "encode" (compress, project) the 3D representation of the 2-sphere, (x,y,z), to a
2D representation (zenith, azimuth).  Similarly, one can "decode" (reconstruct) the polar
representation into the Cartesian representation.  


## Variational AutoEncoders (VAEs)
VAEs can be used to generate images, or to supplement reinforcement learning methods.

### Probabilistic Model Perspective
A VAE can be considered a generative process (probability model) that emits observables, x, based on unobservable 
internal changes (i.e., latent variables), z.  (I'm mixing in some QM perspective here.)

The emission (generative process) of an observable, x[i], is stochastically dependent on the internal state, z. This dependence
is called the likelihood, and is written: x[i] ~ p(x|z).
A particular internal state (set of latent variables), z[i], follow a probability distribution, p(z), called the prior: z[i] ~ p(z).  
The model can be written as a joint probabibility distribution over observables (data) and internal states 
(latent variables): p(x,z) = p(x|z)p(z).

We can measure the observables, but not the internal state. We see a rock fall and the moon in the sky,
and we think: "Didn't Newton develop a single 1-inch equation that describes both these things?"  

From a data perspective, we have the input data (observables) and want to know 
how to compress the data (reduce its dimensionality) without losing too much information.  Our assumption
is that the data's dimensionality can be squeezed quite a bit --- that, like the 2-sphere in Cartesian 
coordinates, the current representation includes informational redundancies. 
In practical terms, these redundancies require more data storage and larger bandwidth.

So the goal is to infer the internal state (latent variables) given what we know about the observables 
(input data). That is, we want to calculate the posterior using Baye's Theorem: p(z|x) = p(x|z)p(z)/p(x).  

...

Anyway, the "encoder" is called an "inference network" in this lingo, and it is used to find the best parametrization, L,
of the approximate posterior, q(z|x,L; W), where z is the latent variable, x is the input, L is
the parametrization (of the chosen distribution family, e.g., mean and stdDev for a normal distribution), and W 
represents the inference network parameters (i.e., neural network weights and biases for the encoder).

The "decoder" is called a "generative network."

For more info mapping between probabilistic modeling lingo and neural network lingo,
re-read "[What is a Variational Autoencoder?](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/)"


#### Kullback-Leibler Divergence
The [KL Divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
measures how different one probability distribution is from another expected distribution.  

Basically, we often approximate stochastic processes as drawing from a familiar probability 
distribution, like a Normal or Log-Normal distribution.  For example, if an empirical
distribution looks normal, then it can be quite informative to discuss two parameters 
(mean and standard deviation) rather than all the specific values and nuances in the data set.
It might also be possible to infer certain results about a normal-looking data set by 
assuming its normality.  However, what do we lose by making these assumptions?

The KL divergence, D(p,q), can help inform us about how much information is discarded by using
a familiar distribution to approximate an empirical one.  

D(p,q) = E{log(p[i]/q[i])} = SUM{ p[i] * (log(p[i]) - log(q[i])) }

It is an error metric.  Interestingly,
it is not a distance metric because it is not symmetric: D(p,q) != D(q,p).

By simultaneously minimizing the KL divergence and number of model parameters, one attempt
to find a faithful, simple model for a data set.  For example, if a 3-parameter model
has just as good a KL divergence as a 6-parameter model, its likely that the 3-parameter
model should be used.  But what if a 2-parameter model has just a slightly bigger KL
divergence?  Careful thought would help here.  If one has enough data, it'd probably be
good to compute on multiple subsets and see what holds up.  ...

#### ELBO: Evidence Lower BOund
The KL divergence is not directly computable.  However, one can bound it by the ELBO.
Then, by optimizing the ELBO, one has optimized the KL divergece...

Again, can't stress how good this article is:  [What is a Variational Autoencoder?](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/)

I twould be nice to re-visit it multiple times, and even extend it (e.g., w/ quantum mechanical lingo, or DSP).



### References
* 2013: Kingma & Welling: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
* 2014: Rezende et al: [Stochastic Back Propagation and Approximate Inference in Deep Generative Models](https://arxiv.org/abs/1401.4082)

### Some Links
* [What is a Variational Autoencoder?](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/)
  - bridges language gap between neural networks and probabilistic models
  - worth reading a couple more times, then building further bridges to quantum mechanical, digital signal processing, and/or manifold lingo
* [Kullback-Leibler Divergence Explained](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained)
  - discusses KL divergence
* [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908)




