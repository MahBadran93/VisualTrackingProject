import numpy as np 
import matplotlib.pyplot as plt 
import cv2
import os
from skimage.filters import window
from skimage.transform import AffineTransform, warp

class MOSSE:
    '''
        - Preprocessing step is performed on every frame of the video before
        initialization or tracking(before using MOSSE Filter).
        
    '''
    def __init__(self):
        self.template = [] 
        self.trackingWindow = []
        self.bBox_Pose = []
        self.Ai = 0
        self.Bi = 0
        
    # crop the target object from an input image and return the croped template
    def CreateTemplate(self, inputImage): 
        '''
        Parameters
        ----------
        inputImage : color image (frame from video )
            input image frame extracted from video to crop a template from
        Returns
        -------
        template : cropped Image  
            croped image of the object we want to track ( ground truth ).
            
        bBox_Pose: croped ROI rectangle coordinates 
        '''
        
        # croped ROI rectangle coordinates 
        bBox_Pose = cv2.selectROI('test', inputImage, showCrosshair = True)
        
        self.template = inputImage[int(bBox_Pose[1]):int(bBox_Pose[1]+bBox_Pose[3]),
                              int(bBox_Pose[0]):int(bBox_Pose[0]+bBox_Pose[2])]
        
        self.template = np.array(self.template).astype(np.int64)
        return self.template, bBox_Pose 
    
    # return the 4 point poistions of the croped template w.r.t the input image
    def getCropedTemplate4Points(self, inputImage): 
        bBox = cv2.selectROI('test', inputImage, showCrosshair = True)
        return bBox
       
    # Log transformation
    # log function is used to enhance low contrast regions
    # and provides more edges for the tracker
    def logTransformTemplate(self, template):
        const = 255 / np.log(1 + np.max(template)) 
        logTransformedImage = const * (np.log(template + 1)) 
        #logTransformedImage = np.divide(np.subtract(logTransformedImage,np.mean(logTransformedImage)),np.linalg.norm(logTransformedImage))
        return logTransformedImage
    
    
    def Normalize(self, template):
        # normalize the template (between 0 and 1) to reduce the lighting 
        # and solve  any contrast changing issues that could affect the tracking
        template = template / 255 
    
        # centering around the mean
        template = template - np.mean(template)
        
        return template
        
    # Apply cosine window to minimize the edge effect in the template 
    # during tracking 
    def CosWindow(self, template):
        cosImage = template * window('hann', template.shape)
        return cosImage
    
        
   # Frequency Domain (DFT)
    def FrequencyDomainTransform(self, template):
        shiftTemplate = np.fft.fftshift(np.fft.fft2(template))
        template = np.abs(shiftTemplate)
        return template
       
    # Affine preprocessing 
    def Affine(self, template): 
        a = -180 / 16
        b = 180 / 16
        r = a + (b - a) * np.random.uniform()
        # rotate the image...
        matrix_rot = cv2.getRotationMatrix2D((template.shape[1]/2, template.shape[0]/2), r, 1)
        img_rot = cv2.warpAffine(np.uint8(template * 255), matrix_rot, (template.shape[1], template.shape[0]))
        img_rot = img_rot.astype(np.float32) / 255
        return img_rot
    #....................................................................
    
    '''
    # Find A gaussian peak image using the input image and the template where 
    # the target object is centered in the image.
    '''
    def findSynthaticGaussianPeak(self, inputImage, template, segma = 100):
        # find center of template (GT) w.r.t the input image
        x_c = template[0] + (template[2] / 2)
        y_c = template[1] + (template[3] / 2)
      
        
        width, height = inputImage.shape
        
        # to vectorise the operation we use mesh grid.
        width = np.arange(width) # give us a numpy array from 1 -> width
        height = np.arange(height) # give us a numpy array from 1 -> height
        array1, array2 = np.meshgrid(width, height) # array1: array of row index values, array2: array of column index values
        
        # Gaussian image
        gaussianArray = (np.exp(-((np.square(array1 - x_c) + np.square(array2 - y_c)) / (2 * segma))))
        # Gaussian image with gaussian peak centered on the object (g). a ground truth 
        gaussianArray = gaussianArray[int(template[1]):int(template[1]+template[3]),
                              int(template[0]):int(template[0]+template[2])]
        return gaussianArray
    
    #.............. Create the dataset.................................. 
    '''
    template: croped image of the target object.
    templateF: the template in frequency domain.
    g: the gussian peak image. 
    G: g in frequency domain.
    '''
    #.....................................................................
    
    def ceateDataSet(self, frame):
        #a,b,c,d = self.getCropedTemplate4Points(frame) # coordinates of croped template
        # input template (object in center ) with log transformation added.

        template, bBox_Pose = self.CreateTemplate(frame)
        # gaussian peak centered template
        g = self.findSynthaticGaussianPeak(frame, self.getCropedTemplate4Points(frame))
        # frequency domain gaussian template
        G = self.FrequencyDomainTransform(g)
        
        template = self.logTransformTemplate(template)
        #template = self.pre_process(template)
        # input template frequency domain 
        templateFreq = self.FrequencyDomainTransform(template) 

        # get template 4 point positions (x, x+x0, y, y+y0)
        return template,templateFreq, g.astype(np.int64), G, bBox_Pose
    
    #.......................................................................
    
    def FindPSR(self, peakresponse, maxVal):
        PSR = (maxVal - np.mean(peakresponse)) / np.std(peakresponse)
        return PSR
        

    def InitializeMOSSE(self, template, G, bBox_Pose, iteration = 9,
                        learning_Rate= 0.125 , initFlag = 0 , A_i=0, B_i=0):
        '''
        Parameters
        ----------
        template : Log of Cropped Image. 
        
        G : Gaussian Peak image in Fourier space (GT).
        
        iteration: number of iterations to train the filter.
        
        learning rate: allows the filter to quickly adapt to appearance
        changes while still maintaining a robust filter (from the paper),
        ranges from [0.01,0.15]
        
        initFlag: a flag value to know if we are in the first frame or not.
        
        A_i, B_i : Updated Ai,Bi in each frame

        Returns
        -------
        Ai, Bi : MOSSEE Filter parameter values 

        '''
        # Resize the G (peak image) to be the same as templateFreq
        G = cv2.resize(G, dsize=(bBox_Pose[2], bBox_Pose[3]), interpolation=cv2.INTER_CUBIC)
        
        '''preprocessing methods here'''
        template = cv2.resize(template, (bBox_Pose[2], bBox_Pose[3]))
        template = self.logTransformTemplate(template)
        template = self.Normalize(template)
        template = self.CosWindow(template)
        
        # template to Frequency Domain
        templateFreq = self.FrequencyDomainTransform(template)
        
 
        # For the first frame we iterate to initialize the Filter H
        if initFlag == 0:  
            
            # if we are in the first frame we initilize the MOSSE filter.
            # Applying the H filter equations presented in the paper
            Ai = G * np.conjugate(templateFreq)
            Bi = (templateFreq * np.conjugate(templateFreq))
        
            for i in range(iteration):
                # preprocess again to make data set from the first frame 
                # with affine affect 
                templateAffine= self.Affine(template)
                template = self.logTransformTemplate(templateAffine)
                template = self.Normalize(template)
                template = self.CosWindow(template) 
                templateFreq = self.FrequencyDomainTransform(template)
                Ai = Ai + (G * np.conjugate(templateFreq))
                Bi = Bi + (templateFreq * np.conjugate(templateFreq))
                
            # initial values of the Filter parameters     
            Ai = learning_Rate * Ai
            Bi = learning_Rate * Bi
        else: 
            # From the second frame till the end, Apply the parameters equations with the learning rate for each frame
            Ai = ((1-learning_Rate) * A_i) + (learning_Rate)*(G * np.conjugate(templateFreq))
            Bi = ((1-learning_Rate) * B_i) + (learning_Rate)*(templateFreq * np.conjugate(templateFreq))
            
        return Ai, Bi 
    
    # read frames instead of video 
    def readFrames(self,path):    
        images = []
        path = path
        for filename in os.listdir(path):
            img = cv2.imread(os.path.join(path,filename))
            if img is not None:
                images.append(img)
        return images
    
    def trackVideo(self, path): 

        # initialize  trcking window position.
       
        # get input video 
        vcap = cv2.VideoCapture(path)
        
        
        # counter for the video frames
        i = 0
        # Iterate over all the frames in the input video 
        while(True):
            # read the current frame 
            ret, curr_frame = vcap.read()
            # convert to gray 
            curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
            # in the first frame, the tracking window will be in the same position
            # as the croped template. 
            # Initialize the filter in the first frame.
            if i == 0: 
                # create the training data. extract template, gaussian peak.
                # In time(template,g) and frequency domain(templateF,G)
                template,templateF,g,G, bBox_Pose = self.ceateDataSet(curr_frame_gray)
                # current frame tracking window coordinates
                self.bBox_Pose = np.array(bBox_Pose).astype(np.int64)
                self.trackingWindow = np.array([bBox_Pose[0],
                                  bBox_Pose[1],
                                  bBox_Pose[0]+bBox_Pose[2],
                                  bBox_Pose[1]+bBox_Pose[3]]).astype(np.int64)
        
                #initialize the filter(using only the first frame).
                # Eq5 in the paper.
                self.Ai,self.Bi = self.InitializeMOSSE(template, G, bBox_Pose)
                i = 1
            else: 
                # Regularizer
                reg = 0.01
                # Filter which will be updated in each frame 
                Hi = self.Ai/(self.Bi + reg)
                              
                # Draw a rectangle bounding box in the current frame which 
                # follows the moving target(by updating the tracking window)
                curr_template = curr_frame_gray[self.trackingWindow[1]:self.trackingWindow[3],
                                    self.trackingWindow[0]:self.trackingWindow[2]]

                
                ''' change preprocessing methods here'''
                curr_template = cv2.resize(curr_template, (self.bBox_Pose[2], self.bBox_Pose[3]))
               

                # Currrent template in Fourier Space
                curr_templateF = self.FrequencyDomainTransform(curr_template)
                
                # The multiplication of the updated Filter H conjugate and curr_templateF
                # results in a peak image (G) which indicates the new location
                # of the target object.we are going to call it newObjLoction_G
                newObjLoction_G = np.conjugate(Hi) * curr_templateF

                '''
                if(i>19):
                    plt.imshow(Hi)
                    plt.show() 
                    plt.imshow(newObjLoction_G)
                    plt.show()   
                '''
                
                # find the position where the peak is, the center(where we will find the target object)
                max_value = np.max(newObjLoction_G)
                max_pos = np.where(newObjLoction_G == max_value)

                #Find the change in the x direction and y direction to update the tracking window
                changeIn_Y = int(np.mean(max_pos[0]) - (newObjLoction_G.shape[0] * 0.5))
                changeIn_X = int(np.mean(max_pos[1]) - (newObjLoction_G.shape[1] * 0.5))
                
                # update the position...
                self.bBox_Pose[0] = self.bBox_Pose[0] + changeIn_X
                self.bBox_Pose[1] = self.bBox_Pose[1] + changeIn_Y

                # trying to get the clipped position [xmin, ymin, xmax, ymax]
                self.trackingWindow[0] = self.bBox_Pose[0] 
                self.trackingWindow[1] = self.bBox_Pose[1] 
                self.trackingWindow[2] = self.bBox_Pose[0]+self.bBox_Pose[2] 
                self.trackingWindow[3] = self.bBox_Pose[1]+self.bBox_Pose[3] 
                self.trackingWindow = self.trackingWindow.astype(np.int64)

                # get the current template using the updated tracking winow..
                curr_template = curr_frame_gray[self.trackingWindow[1]:self.trackingWindow[3], self.trackingWindow[0]:self.trackingWindow[2]]
               
                '''preprocessing methods here'''
                curr_template = cv2.resize(curr_template, (self.bBox_Pose[2], self.bBox_Pose[3]))
                curr_template = self.logTransformTemplate(curr_template)
                curr_template = self.Normalize(curr_template)
                curr_template = self.CosWindow(curr_template)


                # Update A and B
                self.Ai,self.Bi = self.InitializeMOSSE(curr_template, G,bBox_Pose, iteration=0,initFlag=1)
                

            # Visulize the tracking window 
            cv2.rectangle(curr_frame, (self.trackingWindow[0], self.trackingWindow[1]),
                          (self.trackingWindow[2], self.trackingWindow[3]), (255, 0, 0), 2)
            cv2.imshow('test', curr_frame)
            cv2.waitKey(50)
            plt.imshow(curr_frame)
            plt.show()
            # counter 
            i = i + 1 
            

    def trackFrames(self, path): 

       # initialize  trcking window position.

       # counter for the video frames
       i = 0
       # list of frames 
       listOfFrames = self.readFrames(path)
       
       # PSR Evaluation 
       listFramesPSR = []
       
       # Iterate over all the frames in the input video 
       for i in range(len(listOfFrames)):
           
           # read the current frame 
           curr_frame = listOfFrames[i] 
           
           # convert to gray 
           curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
       
           # in the first frame, the tracking window will be in the same position
           # as the croped template. 
           # Initialize the filter in the first frame.
           if i == 0: 
               # create the training data. extract template, gaussian peak.
               # In time(template,g) and frequency domain(templateF,G)
               template,templateF,g,G, bBox_Pose = self.ceateDataSet(curr_frame_gray)
               
               # current frame tracking window coordinates
               self.bBox_Pose = np.array(bBox_Pose).astype(np.int64)
               self.trackingWindow = np.array([bBox_Pose[0],
                                 bBox_Pose[1],
                                 bBox_Pose[0]+bBox_Pose[2],
                                 bBox_Pose[1]+bBox_Pose[3]]).astype(np.int64)
       
               #initialize the filter(using only the first frame).
               # Eq5 in the paper.
               self.Ai,self.Bi = self.InitializeMOSSE(template, G, bBox_Pose)
              
           else: 
               # Regulizer
               reg = 0.01
               
               # Filter which will be updated in each frame 
               Hi = self.Ai/(self.Bi + reg)
    
                             
               # Draw a rectangle bounding box in the current frame which 
               # follows the moving target(by updating the tracking window)
               curr_template = curr_frame_gray[self.trackingWindow[1]:self.trackingWindow[3],
                                   self.trackingWindow[0]:self.trackingWindow[2]]


               
               '''preprocessing methods here'''
               curr_template = cv2.resize(curr_template, (self.bBox_Pose[2], self.bBox_Pose[3]))
               curr_template = self.logTransformTemplate(curr_template)
               curr_template = self.Normalize(curr_template)
               curr_template = self.CosWindow(curr_template)

               
               # Currrent template in Fourier Space
               curr_templateF = self.FrequencyDomainTransform(curr_template)

               
               # The multiplication of the updated Filter H conjugate and curr_templateF
               # results in a peak image (G) which indicates the new location
               # of the target object.we are going to call it newObjLoction_G
               newObjLoction_G = np.conjugate(Hi) * curr_templateF

                      
               # find the position where the peak is, the center(where we will find the target object)
               max_value = np.max(newObjLoction_G)
               max_pos = np.where(newObjLoction_G == max_value)
               
               # PSR evaluation 
               PSR = self.FindPSR(newObjLoction_G, max_value)
               
               if i < 50: 
                   listFramesPSR.append([i, PSR])
               if i == 50:
                   listFramesPSR = np.asarray(listFramesPSR)
                   plt.plot(listFramesPSR[:,0], listFramesPSR[:,1])
                   plt.title('PSR Evaluation, tested with 0.125 LR and 0.01 regularizer ')
                   plt.xlabel('Frames')
                   plt.ylabel('PSR values')

                   plt.show()
                   
               changeIn_Y = int(np.mean(max_pos[0]) - (newObjLoction_G.shape[0] * 0.5))
               changeIn_X = int(np.mean(max_pos[1]) - (newObjLoction_G.shape[1] * 0.5))
               
               # update the position...
               self.bBox_Pose[0] = self.bBox_Pose[0] + changeIn_X
               self.bBox_Pose[1] = self.bBox_Pose[1] + changeIn_Y

               # trying to get the clipped position [xmin, ymin, xmax, ymax]
               self.trackingWindow[0] = self.bBox_Pose[0] 
               self.trackingWindow[1] = self.bBox_Pose[1] 
               self.trackingWindow[2] = self.bBox_Pose[0]+self.bBox_Pose[2] 
               self.trackingWindow[3] = self.bBox_Pose[1]+self.bBox_Pose[3] 
               self.trackingWindow = self.trackingWindow.astype(np.int64)

               # get the current template using the updated tracking winow..
               curr_template = curr_frame_gray[self.trackingWindow[1]:self.trackingWindow[3], self.trackingWindow[0]:self.trackingWindow[2]]
               
               # Update A, B 
               self.Ai,self.Bi = self.InitializeMOSSE(curr_template, G, bBox_Pose, 
                                                      iteration=0,initFlag=1, A_i=self.Ai, B_i= self.Bi)

            
           # Visulize the tracking window 
           cv2.rectangle(curr_frame, (self.trackingWindow[0], self.trackingWindow[1]),
                      (self.trackingWindow[2], self.trackingWindow[3]), (255, 0, 0), 2)
           cv2.imshow('test', curr_frame)
           cv2.waitKey(130)
        
           # counter 
           i = i + 1 
       
       # Plot the PSR evaluation for the first 50 frames 
       print(listFramesPSR)
       cv2.destroyAllWindows() 
        
            
            

#................................Test.......................................              
start = MOSSE()  
# test data 1 with images
# your data path    
path = './Data/test5'      
start.trackFrames(path)

# test data video 
# your video path 
#path = './Data/testVideo.avi'
#path = './Data/test3.mp4'
#start.trackVideo(path)
