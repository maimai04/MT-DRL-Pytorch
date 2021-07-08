# Folder: model

## Overview
This folder contains the following python (.py) files, which contain the following classes and functions:

```
CustomActorCriticNets.py                  : file where all the network architectures and the support functions for
                                            weights are defined. 
    class FeatureExtractorNet             : subclass of nn.Module that implements a feature extractor that is 
                                            shared among actor and critic. This part of the whole net architecture 
                                            takes the original observations as inputs and outputs a certain feature
                                            embedding. This same feature embedding is then passed to the actor and the 
                                            critic networks separately.
    class ActorNet                        : subclass of nn.Module that implements the actor part of the whole architecture.
                                            It takes as input the feature embeddings and outputs a certain number of 
                                            actions (interpreted as mean actions). 
    class CriticNet                       : subclass of nn.Module that implements the critic part of the network architecture.
                                            It takes as input the feature embeddings and outputs one single value, 
                                            interpreted as the state value estimate V.
    class BrainActorCritic                : subclass of nn.Module that combines the 3 networks above (fetaure extractor, 
                                            actor and critic) in one model. In this class, it is also specified
                                            what happend when we predict, evaluate a state, etc., which is a bit different
                                            than normally done in a standard network architecture, hence it had to be customized.
                                            It is also the "Brain" that is then provided to the ppo agent model
                                            (specified in CustomPPOAlgorithm).
  
    init_weights_feature_extractor_net()  : function to initialize (orthogonal) weights for the feature extractor net
    init_weights_actor_net()              : function to initialize (orthogonal) weights for the actor net
    init_weights_critic_net()             : function to initialize (orthogonal) weights for the critic net
```

```
CustomOnPolicyBuffer.py                   : 
    class OnPolicyBuffer                  : class that implements the so-called "buffer", in RL terminology,
                                            where the trajectories / exoeriences collected by the agent are stored
                                            and prepared for the actual training. 
                                            Theoretically one could also go without an extra class for that, or implement 
                                            it just with a function in the ppo agent class, etc., but it is quite practical
                                            to have a class for that so the code is less messy and the buffer can also
                                            be queried easily for debugging purposes and so on.
```

```
CustomPPOAlgorithm.py                      : 
    class PPO_algorithm                    : class that implements the actual ppo algorithm, based on the pseudocode
                                             of the ppo paper (and other sources because honestly that wasn't enough)
                                             This class needs to be passed other instantiated classes in oder to work:
                                                environment class (with all variables and data provided)
                                                brain class (see above, with all variables provided)
                                                buffer class (see above)
```
