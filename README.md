## Aritificial intelligence framework for [Pyrat for the IMT Atlantique course PyRat](https://github.com/vgripon/PyRat)
IMTA 2019-2020

### Dependencies :
- gym
- pygame 
- numpy
---
### Objectives :
- Create a simple reinforcement learning framework for Pyrat. A priori based on  [the OpenAI Gym framework](http://gym.openai.com/) .


### Possible improvements (See issues for more details) :

- Code the game to implement mud.
- Code a replaying system.
- Code the .render() method of the PyratEnv class. Possibly change the render engine (it's Pygame for now) for something more modern.  
Maybe Pyglet ?
[There's a 2D rendering framework included in Gym](https://github.com/openai/gym/blob/master/gym/envs/classic_control/rendering.py) 
- Change the structure of the code to make it more maintainable, easy to use and modular.  
One solution could be to make the environment class and others into python packages ?  
Make more abstract classes ?
- Improve the saving code for the agents and the maze.  
Might be worth it to look into the *design pattern Memento*.


Open to any suggestions, hit me up :  
ngoc-minh-tri.truong@imt-atlantique.net 
