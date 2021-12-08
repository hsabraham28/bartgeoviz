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
    print(stationlocs)

    vizpic = display.Image("./BARTtracksmap.png")
    picdata = image.imread('./BARTtracksmap.png')
    #picdata

    x = stationlocs['Station_Name'].tolist()


    y = 'M10'


    # In[7]:


    for i in x:
        if y in i:
            print(i)


    testlog = pd.read_csv("testocclog.csv")
    print(testlog.columns)
    testlog


    testlog = testlog.drop(columns=['BPD', 'Ref'])


    print(testlog)


    stationarr = (stationlocs['Station_Name']).tolist()
    sarr = []
    for s in stationarr:
        s = s[0:3]
        sarr.append(s)


    stationlocs['Location'] = sarr



    print(stationlocs)


    justxandy = stationlocs.drop(columns=['Station_Name', 'Rain_Critical', 'Asset_Location'])


    print(justxandy)



    merged = testlog.merge(justxandy.set_index('Location'), on='Location')



    merged['dupped'] = merged.duplicated(subset=['Log'])
    merged = merged[merged.dupped != True]


    merged


    vizdata = merged.drop(columns=['dupped'])
    vizdata = pd.read_csv('occ-log-classification/csv/labeled.csv').head(60)
    vizdata['Time'] = vizdata['Time'].apply(lambda x: int(x))

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
    vizdata

    #vizpic


    # fig = plt.figure(figsize=(8, 6), dpi=900) uncomment
    # plt.imshow(picdata) uncomment



    # size = fig.get_size_inches()*fig.dpi uncomment


    # size in pixel
    #plt.plot(1200* .18199, 1100 * .58303, marker='v', color="blue")
    # print(size)
    # print(size[0]* .18199)

    #first get one dot superimposed based on percentages
    #get all the dots on the picture
    #make time attribute figure out time scroll (maybe ipywidgets)

    #uncomment
    # for index,row in vizdata.iterrows():
    #     plt.plot(1200* (row['X_Percentage']/100), 1100 * (row['Y_Percentage']/100), marker='v', color="blue")



    # plt.show()   uncomment

    #x = np.linspace(0, 2 * np.pi)
    # fig = plt.figure(figsize=(8, 6), dpi=150)
    # plt.imshow(picdata)
    # for index,row in vizdata.iterrows():
    #     plt.plot(1200* (row['X_Percentage']/100), 1100 * (row['Y_Percentage']/100), marker='v', color="blue")


    def update(time = 622):
        #line.set_ydata(np.sin(w * x))
    #     def addPoint(scat, new_point):
    #         old_off = scat.get_offsets()
    #         new_off = np.concatenate([old_off,np.array(new_point, ndmin=2)])
            

    #         scat.set_offsets(new_off)
            

    #         scat.axes.figure.canvas.draw_idle()
        fig = plt.figure(figsize=(10, 8), dpi=200)
        plt.imshow(picdata)
        for index,row in vizdata.iterrows():
            if row['Time'] <= time:
                plt.plot(1200* (row['X_Percentage']/100), 1100 * (row['Y_Percentage']/100), marker='v', color="blue")
                #addPoint(fig, [1200* (row['X_Percentage']/100),1100 * (row['Y_Percentage']/100)])
        #############
    #     norm = plt.Normalize(1,4)
    #     cmap = plt.cm.RdYlGn
    #     fi, ax = plt.subplots()
    #     annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
    #                     bbox=dict(boxstyle="round", fc="w"),
    #                     arrowprops=dict(arrowstyle="->"))
    #     annot.set_visible(False)
        
        

    #     def update_annot(ind):

    #         pos = fig.get_offsets()[ind["ind"][0]]
    #         annot.xy = pos
    #         text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))), 
    #                                " ".join([names[n] for n in ind["ind"]]))
    #         annot.set_text(text)
    #         annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
    #         annot.get_bbox_patch().set_alpha(0.4)


    #     def hover(event):
    #         vis = annot.get_visible()
    #         if event.inaxes == ax:
    #             cont, ind = fig.contains(event)
    #             if cont:
    #                 update_annot(ind)
    #                 annot.set_visible(True)
    #                 fi.canvas.draw_idle()
    #             else:
    #                 if vis:
    #                     annot.set_visible(False)
    #                     fi.canvas.draw_idle()

    #     fi.canvas.mpl_connect("motion_notify_event", hover)
        ############
        fig.canvas.draw_idle()
        #plt.show()

    #interact(update);
    interact(update, time = widgets.IntSlider(value=622, min=100, max=2200, step=1));




    import plotly.express as px
    swimscatter = px.scatter(vizdata, y="class", x='Time', color="class", symbol="class", hover_data=['Log', 'Location'])
    swimscatter.update_traces(marker_size=20)
    swimscatter.update_xaxes(categoryorder='category ascending', showgrid=False)
    swimscatter.show()


    # In[31]:


    #swimscatter.write_html(r"C:\Users\Hannah Abraham\Desktop\BARTvisualizations\bart-daily-ops\src\swimscatter.html")


    # In[70]:



    # import plotly.graph_objects as go

    # swim = go.Figure(go.Waterfall(
    #     name = "DailyOps", orientation = "h",
    #     y = catvizdata['Category'],
    #     x = timeaxis,
    #     connector = {"mode":"between", "line":{"width":0, "color":"rgb(0, 0, 0)", "dash":"solid"}}
    # ))

    # swim.update_layout(title = "BART Daily Operations By Category")

    # swim.show()


    # In[72]:


    cattonum = []
    for i in catvizdata['class']:
        cattonum = np.append(cattonum, cattypes.index(i)+1)
    #     if i=='Breakdown':
    #         cattonum = np.append(cattonum, 2)
    #     if i=='Detour':
    #         cattonum = np.append(cattonum, 3)
    #     if i=='Misc':
    #         cattonum = np.append(cattonum, 4)
    cattonum


    # In[38]:


    # pltp.barh("Category", "Time", data = catvizdata, color = "blue")
    # pltp.xlabel("Time")
    # pltp.ylabel("Category")
    # pltp.title("Incidents by Category")
    # pltp.show()


    # plt.xlim(0, 900)
    # #plt.ylim(1, 4)
    # plt.yticks(cattypes)
    # plt.plot(800,2,'r+') 
    # plt.show()

    # Data set
    # height = range(0,900)
    # bars = cattypes
    # y_pos = np.arange(len(bars))

    # # Basic plot
    # plt.bar(height, y_pos, color=(0.2, 0.4, 0.6, 0.6))
     
    # # Custom ticks
    # plt.tick_params(axis='x', colors='red', direction='out', length=13, width=3)

    # #Show the graph
    # plt.show()


    # In[74]:


    # import plotly.express as px
    # import plotly.graph_objects as go

    # years = str(catvizdata['Time'])

    # fig = go.FigureWidget()
    # #f = go.FigureWidget(fig)
    # refnums = catvizdata['Reference']


    # fig.add_trace(go.Bar(x=years, y=([0.5] * catvizdata.shape[0]),
    #                 base=cattonum,
    #                 marker_color='blue',
    #                 name='incidents'))
    # fig.update_yaxes(nticks=8)
    # fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightBlue')
    # fig.update_xaxes(tickwidth=2)
    # fig.update_layout(
    #     yaxis = dict(
    #         tickmode = 'array',
    #         tickvals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #         ticktext = cattypes
    #     )
    # )
    # for data in fig.data:
    #     data["width"] = 3 #Change this value for bar widths
        
    # out = Output()
    # @out.capture(clear_output=True)
    # # create our callback function
    # def showdetails(trace, points, selector):
    #     for i in points.point_inds:
    #         print(points.point_inds)
    #         print(points)
    #         print(trace)
    #         print(selector)
            

    # l = fig.data[0]
    # l.on_click(showdetails)

    # #fig.show()
    # VBox([fig, out])


    # # In[40]:


    # long = px.bar(catvizdata, x="Time", y="Category", color="Category", title="Long-Form Input")
    # long.show()

if __name__ == "__main__":
    app.run()
