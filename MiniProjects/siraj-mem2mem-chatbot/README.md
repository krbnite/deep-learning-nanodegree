

The goal is to build a chatbot that can answer any question you give it.  
We will attempt to do this using [Keras](https://keras.io/).

## Motivation
Google is great, but when you're wondering about the tail region of the
magnetosphere, an expert chatbot might be better!  Research papers are great:
they document the results and progress of a particular individual or team
in a given area of inquiry.  However, anyone who has spent time mastering
a particular subject matter, reading through and dissecting a paper's contents
can take the better part of a day (or worse), especially when new to the field.
And this is sometimes before you decide that the paper did not really answer
your questions like you thought it would.  

Wouldn't it be great to just have an expert chatbot that could read through the paper,
summarize it, and answer all your questions?!  At the least, this could help prioritize
which papers to focus on and weed out ones that are not worth the time investment.

Domain-specific chatbots are within the realm of possibility.  

## Data
Did Siraj make it clear where he got the data in the video?  Not sure... I got it 
from [research.fb.com](https://research.fb.com/downloads/babi/).

Oh, in the links, there is a link to the 
[how\_to\_make\_a\_chatbot](https://github.com/llSourcell/How_to_make_a_chatbot) challenge.

## GRU Notes
u[i] = sigmoid(W[u]x[i] + U[u]h[i-1] + b[u])
r[i] = sigmoid(W[r]x[i] + U[r]h[i-1] + b[r])
g[i] = tanh(W[g]x[i] + r[i]\*U[g]h[i-1] + b[g])
h[i] = u[i]\*g[i] + (1-u[i])\*h[i-1]

![gru-vs-lstm](/assets/GRU-vs-LSTM.png)


## Papers
2015: Weston, Chopra, & Borders: [Memory Networks](https://arxiv.org/pdf/1410.3916.pdf)

## Some Links
* https://yerevann.github.io/2016/02/05/implementing-dynamic-memory-networks/
* http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/
* https://github.com/domluna/memn2n

