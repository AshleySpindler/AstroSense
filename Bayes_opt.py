"""
Bayesian Hyperparameter Optimisation using hyperopt
   - sample hyperparamters from a prior dist, learn best
     inputs by modifying the dist based on the resulting
     loss
"""
from hyperopt import hp, tpe, Trials, fmin, STATUS_OK, plotting, base
from hyperopt.pyll.stochastic import sample

from Autoencoder import Models_TFrecord as Models
import Data.GalZoo2_TFIngest as Data_In

from keras.callbacks import EarlyStopping
from keras import optimizers

import numpy as np
import csv
from timeit import default_timer as timer
from numpy.random import choice, seed
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

def objective(params):
    """Objective function for Hyperparameter Tuning
    
    We build a vae model from .Autoencoder using sampled hyper params
    Train model for limited number of epochs (we don't wanna be here
    all week) and save the params to a csv along with the loss function
    """

    global ITERATION, BEST_LOSS, EPOCHS, STEPS_PER_EPOCH
    
    ITERATION += 1

    #pull hyperparamters out of the dictionary, make them integers if need be
    filters = [int(params['Filters_1']), int(params['Filters_2']), int(params['Filters_3'])]
    latent = int(params['Latent'])
    learning_rate = params['learning_rate']
    beta = params['beta']#1*32/64**2 #not playing with beta yet, as it has the affect of making reconstruction worse
    layers = 3 #not playing with layers, difficult to code and unlikely to improve loss

    #make the model
    encoder, inputs, latent_z = Models.encoder_model(x_train, layers, filters, latent, beta)
    decoder = Models.decoder_model(encoder, layers, filters, latent)
    vae = Models.autoencoder_model(inputs, decoder, latent_z[2])

    #Earlystopping clause, this is mainly so we don't spend a week testing atm
    ES = EarlyStopping(monitor='val_loss', min_delta=0.001,
                       patience=10)

    #time that shit
    start = timer()

    #compile and fit, only new stuff is the callback
    #need to play with verbosity, rn too much clutter in terminal
    vae.compile(optimizers.Adam(lr=learning_rate),
                loss=Models.nll,
                target_tensors=inputs,
                )
    vae.fit(batch_size=None, epochs=EPOCHS,
            steps_per_epoch=STEPS_PER_EPOCH,
            validation_data=(x_valid, None),
            validation_steps=STEPS_PER_EPOCH,
            callbacks=[ES],
            verbose=2,
           )


    run_time = timer()-start

    #calculate the final validation loss
    loss = vae.evaluate(x_valid, x_valid, steps=STEPS_PER_EPOCH)

    #we wanna keep the weights of the best model so we can test it and
    #continue training if we want
    if loss < BEST_LOSS:
        #vae.save('Data/Scratch/GalZoo2/Bayes_Best_Model.h5')
        vae.save_weights('Data/Scratch/GalZoo2/bayes_Best_Weights.h5')
        BEST_LOSS = loss

    # Write to the csv file ('a' means append)
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, params['Filters_1'], params['Filters_2'], params['Filters_3'],
                     params['Latent'], params['learning_rate'], params['beta'], ITERATION, run_time, STATUS_OK])
    of_connection.close()

    return {'loss': loss, 'params': params, 'iteration': ITERATION,
            'train_trime': run_time, 'status' : STATUS_OK}


# Define Hyperparameter space

space = {
         'Filters_1' : hp.quniform('Filters_1', 1, 64, 1),
         'Filters_2' : hp.quniform('Filters_2', 1, 64, 1),
         'Filters_3' : hp.quniform('Filters_3', 1, 64, 1),
         'Latent' : hp.quniform('Latent', 1, 50, 1),
         'learning_rate': hp.loguniform('learning_rate',
                                         np.log(0.00001),
                                         np.log(0.001)),
         'beta' : hp.uniform('beta', 1, 25)
        }

# Algorithm
tpe_algorithm = tpe.suggest

# Trials object to track progress, if you load an old Trials(), you need to increase MAX_EVALS
bayes_trials = Trials()
#bayes_trials = pickle.load(open('Data/Scratch/GalZoo2/Bayes_Trials_database.p', 'rb'))

# OPEN DATA SETS

# tfrecord stuff
train_data = 'Data/Scratch/GalZoo2/galzoo_train.tfrecords'
valid_data = 'Data/Scratch/GalZoo2/galzoo_valid.tfrecords'

x_train = Data_In.create_dataset(train_data, 200, None)
x_valid = Data_In.create_dataset(valid_data, 200, None)

# File to save first results (tail .csv lets us check file while running)
out_file = 'Data/Scratch/GalZoo2/bayes_hyperopt_GalZoo2.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'Filters 1', 'Filters 2', 'Filters 3', 'Latent', 'Learning Rate', 'beta', 'iteration', 'train_time', 'status'])
of_connection.close()

MAX_EVALS = 500 #how many tests we wanna do, balanced is of course time/results

#we need some variables passed betwen functions
global ITERATION, BEST_LOSS, EPOCHS, STEPS_PER_EPOCH

BEST_LOSS = 100
ITERATION = 0
EPOCHS = 100
STEPS_PER_EPOCH = 50

# Optimize, find the smallest loss from fn, with space, using algo
# for MAX_EVALS steps, collate results in bayes_trials
best = fmin(fn = objective, space = space, algo = tpe.suggest, 
            max_evals = MAX_EVALS, trials = bayes_trials)

#make some plots and save our trials object
#we can reload the trials object and keep training
#or just look at the pretty results

plotting.main_plot_history(bayes_trials)
plt.savefig('Data/Scratch/GalZoo2/bayes_trials_main_history.png')

plotting.main_plot_histogram(bayes_trials)
plt.savefig('Data/Scratch/GalZoo2/bayes_trials_main_histogram.png')

plotting.main_plot_vars(bayes_trials)
plt.savefig('Data/Scratch/GalZoo2/bayes_trials_main_plot_vars.png')

pickle.dump(bayes_trials, open('Data/Scratch/GalZoo2/bayes_Trials_database.p', 'wb'))
