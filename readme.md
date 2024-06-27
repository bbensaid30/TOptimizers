# TOptimizers

## Goal of the project
I implement new optimizers in Tensorflow and test them on some real datasets with many initializations.

## Structure

* activations_perso: define personal activation functions, used for code verification.
* classic: redefine classical deterministic optimizers of Tensorflow (Momentum, Adam,...) by modifying the stopping criteria
* data: data preparation
* eval: functions evaluating some metrics for your model and the prediction time
* init: redefine some classical initializations (Xavier, Bengio) since their Tensorflow original versions are not reproductible
* lds: enable to give weights to some part of the data in the objective function
* metrics_perso: define personal metrics to evaluate a model
* model: define the neural networks
* perso: implement some Armijo optimizers (LC:LCEGD1 and LCD:LCEGD2 in the original article). 
The hyperparameters rho and eps_egd could be ignored (it was just an unsuccessful test).
* prepared_eqn: data preparation for a physical problem about state equations (no matter)
* read: read data files in csv format
* tirages: run sequentially or in parallel trainings with different initializations
* training: the general training function that will call one of the optimizer
* utils: useful functions

## Some examples 

* main_MNIST: the MNIST database with fully connected neural networks
* main_FashionMnist: the FASHION MNIST problem with a LeNet1 convolutional network
* main_poly: run the analytical benchmarks. Essential to check the implementations of your own optimizers.
* main_eqn: a exemple on a state equation (goal: predict energy and pression from density and temperature)
* main_Runge: approximate the Runge function by a neural network