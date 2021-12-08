#!/usr/bin/env python
# coding: utf-8

from flask import Flask
import pandas as pd
import numpy as np
from IPython import display
import plotly.graph_objects as go
from matplotlib import image
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import ipywidgets as widgets
from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'notebook')
from ipywidgets import *
import matplotlib.pyplot as pltp
#get_ipython().run_line_magic('matplotlib', 'inline')
from thing import *

app = Flask(__name__)

@app.route("/")
def alldisplays():
    stationlocs = pd.read_csv("BARTLocationPercentages.csv")

    vizpic = display.Image("./BARTtracksmap.png")
    picdata = image.imread('./BARTtracksmap.png')


    vizdata = pd.read_csv('occ-log-classification/csv/labeled.csv').head(60)
    print(vizdata)

    model = really_jank_get_model()

    class_legend = {
        0:"Misc",
        1:'Other BPD',
        2:'Homeless',
        3:'Medical',
        4:'Patron interference',
        5:"Vehicle failure",
        6:"Wayside equipment",
        7:"Software related failures",
        8:"Human Error",
        9:"Weather",
        10:"Info (no error)",
        11:"Delays",
        12:"Track obstruction",
        13:"Schedule maintenance"
    }

    vizdata['class'] = vizdata['Log'].apply(lambda log: np.argmax(model.predict([log]), axis=1)[0])
    vizdata['class'] = vizdata['class'].apply(lambda code: class_legend[code])



    """
    def update(time = 622):

        fig = plt.figure(figsize=(10, 8), dpi=200)
        plt.imshow(picdata)
        for index,row in vizdata.iterrows():
            if row['Time'] <= time:
                plt.plot(1200* (row['X_Percentage']/100), 1100 * (row['Y_Percentage']/100), marker='v', color="blue")

        

        fig.canvas.draw_idle()

    interact(update, time = widgets.IntSlider(value=622, min=100, max=2200, step=1));
    """


    import plotly.express as px
    swimscatter = px.scatter(vizdata, y="class", x='Time', color="class", symbol="class", hover_data=['Log'])
    swimscatter.update_traces(marker_size=20)
    swimscatter.update_xaxes(categoryorder='category ascending', showgrid=False)
    swimscatter.show()


if __name__ == "__main__":
    app.run()
