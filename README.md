# Visual Tracking Project
This software is an attempt to implement of visul tracking filter called Mosse Filter by this paper: <br> 
https://www.cs.colostate.edu/~vision/publications/bolme_cvpr10.pdf <br> 

# Packages: 
- opencv-python <br> 
# To run: <br> 
- Run main.py
- The test video will open, then select a bounding box around the the object you want to track(try to make the bounding box close to your target), use your mouse to select the bounding box. 
Look at this video: <br>
https://www.loom.com/share/27c61f2acf954877b97c3d9aaa32da89 <br> 

It can be noticed the tracking quality is not good sometimes  where it is noticed that the tracking window diverge from the object, by changing the parameters like the learning rate and the regularizer affect the quality and make a better tracking. <br> 
The program is commented and each step is explained. See the PSR evaluation below with specified parameters: <br> 
<p align="center">
  <img src="Resources/PSR1.png" width="700" title="hover text">
</p>
The PSR evaluation can asess the quality of the filter, as the paper says if the values of the PSR between the 20 and 60 it means that the filter is tracking good. 


