from flask import Flask, render_template,request

from ast import arg
from mylib import config, thread
from maip import prog
from mylib.detection import detect_people
from imutils.video import FPS
from scipy.spatial import distance as dist
import numpy as np
import argparse, imutils, cv2, os

videourl=""

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('homepage.html')


@app.route('/', methods = ['GET','POST'])
def getvalue():
    if request.method == 'POST':
        print("post")
    videourl=request.form['video']
    videourl=videourl[1:len(videourl)-1]
    prog(videourl,"")
    print("video completed")
    return render_template('homepage.html')

@app.route("/live",methods =['GET'])
def live_page():
    return render_template('livepage.html')

@app.route('/live', methods = ['GET','POST'])
def getans():
    liveurl=request.form['url']
    prog(videourl,liveurl)
    print("video completed")
    return render_template('homepage.html')

if __name__ == "__main__":
    app.run(debug=True)