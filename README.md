## Projet reinforcement learning sur Pyrat
IMTA 2019-2020

---
### Objectifs:
- Créer un framework compréhensible et simple à prendre en main de reinforcement learning Pyrat. À priori basé sur [le framework OpenAI Gym](http://gym.openai.com/) .


### Amélioration possibles :

- coder le jeu avec de la boue
- coder un système de replay 
- coder la méthode .render() de la classe PyratEnv. Possiblement changer le renderer (pour l'instant Pyrat) pour quelque chose de plus moderne.  
Peut-être pyglet ?
[Il y a un framework intégré dans gym](https://github.com/openai/gym/blob/master/gym/envs/classic_control/rendering.py) 
- Changer la structure du code pour le rendre plus lisible/plus modulable/plus maintenable.  
On pourrait par exemple faire plus de paquets Python importable ?  
Faire des interfaces/classes abstraites ?
- Améliorer les méthodes de save d'agents et de labyrinthes.  
Ça vaudrait peut-être le coup de regarder le *design pattern Memento*.


Ouvert à toutes suggestions hésitez pas.
