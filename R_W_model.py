
import os, random, time, datetime, glob, copy, re, matplotlib
from random import randint, randrange


def Blocking():
	Stims[:,1] = 1;
	Stims[n/2+1:n,2] = 1;
	Response[:] = 1;


def PerfectCue():
	Stims[:,:] = randint(0,1);
	Response[:] = random(n,1) < Stims*[1,0];

def scatterplot(x_data, y_data, x_label, y_label, title):

    # Create the plot object
    _, ax = plt.subplots()

    # Plot the data, set the size (s), color and transparency (alpha)
    # of the points
    ax.scatter(x_data, y_data, s = 30, color = '#539caf', alpha = 0.75)

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


n = 50 # Number of trials
c = 2 # Number of cues
Stims = zeros((n,c)) 
Weights = zeros((c,n + 1))
Outcome = zeros((n,1))
Prediction = zeros((n,1))
Response = zeros((n,1))
e = .3 # Learning rate (free parameter)

case = 2

switcher = {
	1: Blocking,
	2: PerfectCue,
}
switcher.get(case)

#### Simulate ####

for i in range(n):
	Prediction[i] = Stims[i,:]*w[:,i]
	Response[i] = random<p(t);
	pred_error = Response[i] - Prediction[i]
	Weights[:,i + 1] = Weights[:,i] + e*pred_error*Stims[i,:]

scatterplot(range(n), Response, 'time', 'Response', 'Responses over time');

#def R_W_Model():


#def I_S_Model():

