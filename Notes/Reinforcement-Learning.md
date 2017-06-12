


## Markovian Things
Better know your history, right?  Not if you're Markovian. In Markovia, 
everyone is blackout drunk and stumbling around. Nobody remembers anything.
History doesn't matter.

These drunkards aren't just any ol' drunkards: they are stochastic processes.  
A stochastic process is said to
have the Markovian property  when its future state is independent its past states,
given its present.  

A Markov Model is a model that has the Markov property. 

## History Buff Conciliation
Ok, it's not that history doesn't matter in a Markov process.  It's that
you can encode all the relevant history in the current state.  

For example, say a rock is falling in a world without wind, friction, etc.
If we just know its location x[t1] at time t1, we can't say where it will 
be at time t2.  The same goes if we know both x[t1] and its velocity, v[t1].
However, if we know x[t1], v[t1], and its acceleration, g, due to gravity,
then we can say something about where the rock with be at time t2.

## Policy and Reward
In reinforcement learning, they speak a lot about states (or state spaces),
actions (or action spaces), and policies that maximize reward.  It is not 
altogether dissimilar from physics, where we talk about states, configuration 
space, phase spaces, and geodesics.

A policy refers to the action taken for a given state.  A reward is something
attained for making good decisions, which has to do with having a good policy.
The goal is to find a policy that maximizes your reward.  To my mind, this resembles
Lagrangians, Hamiltonians, and the principle of least action.

## How to Figure Out a Good Policy
The method we cover is Q-Learning.  It's pretty brute force: get in state s, try
action a, and record the Q-value.  Since action a probably landed you a new state,
s', try another action a' and record that Q-value.  Bumble around the state space
in this manner for a long time and eventually you'll have a dependable (state, action)
table of Q-values.  At this point, you can define your policy as choosing that action
A that maximizes the Q-value given the state S.

In practice, devising such a table is only feasible for small state spaces.
But most state spaces really aren't that small... And so learning the Q function
in this manner becomes untenable.  

But how well do you really have to know the Q function?  Afterall, we're really
just approximating it.  Can't we just use another approximation technique?

This is where deep learning comes into reinforcement learning, and the answer is,
"Yes we can!"

## Some Links
* [Simple Reinforcement Learning with Tensorflow, Part 0: Q-Learning with Tables and Neural Networks](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0)
* [Demystifying Deep Learning](http://neuro.cs.ut.ee/demystifying-deep-reinforcement-learning/)
* [DRL: From Pong to Pixels (Karpathy)](http://karpathy.github.io/2016/05/31/rl/)
* [The OpenAI Gym](https://blog.openai.com/openai-gym-beta/)

## Some Video
* [David Silver's DRL Lecture](http://videolectures.net/rldm2015_silver_reinforcement_learning/)
* [David Silver's RL Course](http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching.html)
* [UC Berkeley's DRL Course](http://rll.berkeley.edu/deeprlcourse/)
* [Nano de Freitas' ML Course](https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/)

## Some Papers
* Mnih et al (2013): [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
  - Landmark paper from DeepMind
* Mnih et al (2015): [Human-Level Control through Deep Reinforcement Learning](https://scholar.google.com/scholar?cluster=12439121588427761338&hl=en&as_sdt=0,31)





