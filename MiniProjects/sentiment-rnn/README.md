For some reason, the requirements file would have you get regular ol' TensorFlow.  
I thought it was kind of weird, but figured maybe this project didn't need the GPU.

Here's the thing: you need the GPU! If you have it, then use it.  

On my AWS CPU, the training session took 40+ minutes.  Out of curiosity I used another
conda environment which instead used TensorFlow's GPU library.  The very same parameter
set up, etc, took less than 5 minutes to run. Using a larger batch size would probably 
improve this performance even more.

Lessons Learned:  Use your damn GPU!  :-p
 
