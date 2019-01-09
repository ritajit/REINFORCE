# REINFORCE
Using REINFORCE to learn Gym environments -- `LunarLander-v2` and `CartPole-v0`.  
Read the [write-up](https://medium.com/@dey.ritajit/learning-cart-pole-and-lunar-lander-through-reinforce-9191fa21decc).

## Test a Trained Model
Run the program with `REPLACE_MODEL` set to `False`. If there exists a directory named `model` the program will use it to play the environment.  
Set `RENDER` to `True` to enable video.

## Training a New Model
If `model` directory does not exist then the program trains a new model.   
If `model` directory exists and you wish to train a new model, run the program with `REPLACE_MODEL` set to `True`.


