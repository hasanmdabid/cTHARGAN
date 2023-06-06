""""
This Functon will plot the Epoch vs Losses (Discriminator and generator)
1st we will take the average of d_loss_real , d_loss_fake, and g_loss
2nd we will plot them in the graphs
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv('/home/abidhasan/Documents/TimeGAN_Tensorflow_2/Conditional_GAN/results/D_and_G_model_Loss_1K_64Batch_minorityonly_conv1d_lstm.csv', sep=';')

data = data.drop(['gen_activation',' batch_per_epoch'], axis=1)
#data = data.drop(['g_loss'], axis=1)
print(data.head(5))
df = data.groupby(by=[" epochs"]).mean()
print(df.head())


df.plot()
plt.show()
#plt.savefig('HARCGAN_performance_100Epochs.jpg')

