

### __Usage:__
1. [optional] build docker container and set docker_run_example.sh to correct paths (and rename to docker_run.sh)
2. Set hyperparameters in hparams_example.json and rename to hparams.json; both hparams.json and docker_run.sh are in .gitignore
3. execute run.py in suitable python environment or docker_run.sh 
4. [optional] check results in tensorboard  

### __Hyperparameter explanations:__
- *num_epochs*: number of epochs the model is trained
- *threshold*: if recurrent, the threshold that entropy needs to be below of for early output.
- *occlusion_size*: size of the occlusion  used for training (% of noise pixel or length of cutout)
- *steps*: maximum number of recurrent steps. 1 means feedforward only
- *recurrence*: needs to be a list of booleans of length 3. boolean flags wether a block has a lateral (recurrent) connection.
- *residual*: wether or not residual connections are enabled.
- *batch_size*: batch size for training
- *lr_start*: learning rate the lr_scheduler is initialised with 

### __Reference:__  
The basic concept of having recurrent connections to flexibly trade off speed for accuracy and the Bl_model architecture and naming scheme are from the paper [Recurrent neural networks can explain flexible trading of speed and accuracy in biological vision](http://dx.doi.org/10.1371/journal.pcbi.1008215) and the respective [Github](https://github.com/cjspoerer/rcnn-sat)  
  

### __Related:__
The role of feedback in biological vision and therefore their possible importance for the development of computer vision has been described 
in the paper [Beyond the feedforward sweep: feedback computations in the visual cortex](http://dx.doi.org/10.1111/nyas.14320) in a general way 
while several authors have put forth more concrete evidence of narrow effects.  
Tang et. al find in [Recurrent computations for visual pattern completion](http://dx.doi.org/10.1073/pnas.1719397115) that recognition of 
occluded objects needs additional computation as it is impaired by backward masking and has a delayed response time while a common 
feedforwad convolutional neural network (Alexnet) has problems with occlusion. Performance can be improved for this computational model 
by adding recurence to the last fully connected layer.  
Kar and DiCarlo look into "late-solved" images responses in macaque monkeys and find that deactivating the ventral prefrontal cortex leads to worse accuracy in "IT population decode accuracy" while the responses in IT become more similar to those produced by various feedforward neural networks in the paper [Fast recurrent processing via ventral prefrontal cortex is needed by the primate ventral stream for robust core visual object recognition](http://dx.doi.org/10.1101/2020.05.10.086959) 


