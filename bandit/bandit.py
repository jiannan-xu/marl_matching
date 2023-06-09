import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
import math
import pandas as pd
from abc import ABC, abstractmethod

import logging
from abc import (
    ABC,
    abstractmethod,
)
from collections import defaultdict
from typing import List
from uuid import uuid4

logger = logging.getLogger(__name__)

def generate_data(num_cases,num_bandits,mu,sigma):
    '''
    args: 
    num_prices: number of prices for each distribution
    mu: mean of distribution
    sigma: standard deviation of distribution
    Generates prices for a given number of prices, mu, and sigma
    '''
    # generate prices
    random.seed(2023)
    #Generate Data
    bandits = np.arange(0,num_cases)
    for i in range(0,num_bandits):
        prices = stats.truncnorm.rvs(0,np.inf,loc=mu[i],scale = sigma[i],size = num_cases)
        bandits = np.concatenate((bandits,prices),axis = 0)
    print(bandits)
    bandit_prices = pd.DataFrame(bandits.reshape(num_bandits+1,num_cases).T,columns = ['bandit'] + ['price_{}'.format(i) for i in range(0,num_bandits)])
    return bandit_prices

class UCB1(object):
    def __init__(self, df, num_rounds: int = 10000,num_bandits: int = 3, rho: float = 1):
        """
        UCB algorithm for bandit problems.
        Args:
            num_bandits (int): Number of bandits.
            Qt_a (float): average reward of action a prior to the time t.
            Nt_a (int): number of times action a has been selected prior to time t.
            rho (float): exploration parameter.
            mu (float): True mean.
            lower_bound (float): Lower bound for rewards.
            upper_bound (float): Upper bound for rewards.
        """
        # set parameters
        self.df = df
        self.num_rounds = num_rounds
        self.num_bandits = num_bandits
        self.Qt_a = 0
        self.Nt_a = np.zeros(self.num_bandits)
        self.rho = rho
        self.sum_rewards = np.zeros(self.num_bandits) # cumulative sum of reward for a particular bandit
        self.hist_t = [] # holds the natural log of each round
        self.hist_achieved_rewards = [] # holds the history of the UCB CHOSEN cumulative rewards
        self.hist_best_possible_rewards = [] # holds the history of OPTIMAL cumulative rewards
        self.hist_random_choice_rewards = [] # holds the history of RANDONMLY selected actions rewards


    def implement(self):
        """
        Pulling the arm of the bandit and collecting the reward.
        """
        #loop through no of rounds 
        for t in range(0,self.num_rounds):
            UCB_Values = np.zeros(self.num_bandits) # array holding the ucb values where we pick the max  
            action_selected = 0
            for a in range(0, self.num_bandits):
                if (self.Nt_a[a] > 0):
                    log_t = math.log(t) 
                    self.hist_t.append(log_t) # to plot natural log of t
                
                    # calculate the UCB
                    Qt_a = self.sum_rewards[a]/self.Nt_a[a]
                    ucb_value = Qt_a + self.rho*(log_t/self.Nt_a[a]) 
                    UCB_Values[a] = ucb_value
                # if this equals zero, choose as the maximum. Cant divide by negative     
                elif (self.Nt_a[a] == 0):
                    UCB_Values[a] = 1e500 #make large value
        
            #select the max UCB value
            action_selected = np.argmax(UCB_Values)

            #update Values as of round t
            self.Nt_a[action_selected] += 1
            reward = self.df.values[t, action_selected+1]
            self.sum_rewards[action_selected] += reward

            #these are to allow us to perform analysis of our algorithmm
            r_values = self.df.drop(['bandit'],axis = 1).values[t,:]     # get all rewards for time t to a vector
            r_best = r_values[np.argmax(r_values)]      #s elect the best action 
            
            pick_random = random.randrange(self.num_bandits) #choose an action randomly
            r_random = r_values[pick_random] #np.random.choice(r_) #select reward for random action
            if len(self.hist_achieved_rewards)>0:
                self.hist_achieved_rewards.append(self.hist_achieved_rewards[-1]+reward)
                self.hist_best_possible_rewards.append(self.hist_best_possible_rewards[-1]+r_best)
                self.hist_random_choice_rewards.append(self.hist_random_choice_rewards[-1]+r_random)
            else:
                self.hist_achieved_rewards.append(reward)
                self.hist_best_possible_rewards.append(r_best)
                self.hist_random_choice_rewards.append(r_random)

        return self.Nt_a, self.sum_rewards, self.hist_achieved_rewards,self.hist_best_possible_rewards, self.hist_random_choice_rewards

if __name__ == "__main__":
    bandit_prices = generate_data(num_cases = 10000,num_bandits = 10, mu = np.arange(8,18),sigma = np.repeat(2,10))
    print(bandit_prices)
    ucb = UCB1(bandit_prices, num_rounds = len(bandit_prices.index),num_bandits = 10, rho = 1)
    Nt_a,sum_rewards,hist_achieved_rewards,hist_best_possible_rewards, hist_random_choice_rewards = ucb.implement()
    print("Number of times each bandit was selected: ", Nt_a)
    print("Sum of rewards for each bandit: ", sum_rewards)
    print("Cumulative rewards for UCB: ", hist_achieved_rewards[-1])
    print("Cumulative rewards for best possible action: ", hist_best_possible_rewards[-1])
    print("Cumulative rewards for random action: ", hist_random_choice_rewards[-1])

    #plot the results
    figure = plt.figure()
    plt.plot(range(10000), ucb.hist_achieved_rewards[0:10000], label = "UCB")
    plt.plot(range(10000), ucb.hist_random_choice_rewards[0:10000], label = "Random Choice")
    plt.plot(range(10000), ucb.hist_best_possible_rewards[0:10000], label = "Best Possible")
    plt.legend()
    plt.xlabel("Number of Rounds")
    plt.ylabel("Cumulative Rewards")
    plt.title("UCB vs Random Choice vs Best Possible")
    plt.savefig("figures/cumulative_rewards.eps", format='eps')

    #plot the results
    figure = plt.figure()
    plt.plot(range(10000), np.array(ucb.hist_best_possible_rewards[0:10000]) - np.array(ucb.hist_achieved_rewards[0:10000]), label = "UCB - Regret")
    plt.plot(range(10000), np.array(ucb.hist_best_possible_rewards[0:10000]) - np.array(ucb.hist_random_choice_rewards[0:10000]), label = "Random Choice - Regret")
    plt.legend()
    plt.xlabel("Number of Rounds")
    plt.ylabel("Regret")
    plt.title("UCB vs Random Choice vs Best Possible")
    plt.savefig("figures/regret.eps", format='eps')
