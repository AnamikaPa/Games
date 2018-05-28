"""
https://gym.openai.com/envs/Pendulum-v0/
Pendulum-v0

The inverted pendulum swingup problem is a classic problem in the control literature. 
In this version of the problem, the pendulum starts in a random position, and the goal 
is to swing it up so it stays upright.

"""
import gym, random
import numpy as np 
import tflearn
from statistics import mean, median
from collections import Counter
from sklearn import linear_model

LR = 1e-3
env = gym.make('Pendulum-v0')
env.reset()
goal_steps = 500
score_requirement = 50
initial_games = 100

print env.action_space

def initial_population():
    # [OBS, MOVES]
    training_data = []
    # all scores:
    scores = []
    # just the scores that met our threshold:
    accepted_scores = []
    # iterate through however many games we want:
    for _ in range(initial_games):
        score = 0

        # moves specifically from this environment:
        game_memory = []
        
        # previous observation that we saw
        prev_observation = []
        
        # for each frame in 200
        for _ in range(goal_steps):
            # choose random action (0 or 1)
            action = env.action_space.sample()
            # do it!
            observation, reward, done, info = env.step(action)
            
            # notice that the observation is returned FROM the action
            # so we'll store the previous observation here, pairing
            # the prev observation to the action we'll take.
            if len(prev_observation) > 0 :
                print action
                game_memory.append([prev_observation, action])

            prev_observation = observation
            score+=reward
        
            if done: 
            	break

        # IF our score is higher than our threshold, we'd like to save
        # every move we made
        # NOTE the reinforcement methodology here. 
        # all we're doing is reinforcing the score, we're not trying 
        # to influence the machine in any way as to HOW that score is 
        # reached.
        
        accepted_scores.append(score)
        for data in game_memory:
            # saving our training data
            training_data.append([data[0], data[0]])

        # reset env to play again
        env.reset()
        # save overall scores
        scores.append(score)
    
    # just in case you wanted to reference later
    training_data_save = np.array(training_data)
    np.save('saved.npy',training_data_save)
    
    # some stats here, to further illustrate the neural network magic!
    print('Average accepted score:',mean(accepted_scores))
    print('Median score for accepted scores:',median(accepted_scores))
    print(Counter(accepted_scores))
    
    return training_data

#initial_population()

def train_model(training_data, model=False):
    #print training_data
    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]))
    y = [i[1] for i in training_data]

    clf = linear_model.Lasso(alpha=0.1)
    clf.fit(X,y)
   
    return clf

training_data = initial_population()
model = train_model(training_data)

scores = []
choices = []

# Testing
for each_game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(goal_steps):
        env.render()

        if len(prev_obs)==0:
            action = env.action_space.sample()
        else:
            action = model.predict(prev_obs.reshape(-1,len(prev_obs)))[0]
            
        choices.append(action)
                
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation

        game_memory.append([new_observation, action])
        score+=reward
        if done: 
        	break

    scores.append(score)

print('Average Score:',sum(scores)/len(scores))
