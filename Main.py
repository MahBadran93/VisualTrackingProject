import numpy as np 
import matplotlib.pyplot as plt 
import cv2
import os

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

        '''
        
        # selectROI return value : (x , x+x1, y , y+y1) 
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
    def logTransformTemplate(self, template):
        const = 255 / np.log(1 + np.max(template)) 
        logTransformedImage = const * (np.log(template + 1)) 
        #logTransformedImage = np.divide(np.subtract(logTransformedImage,np.mean(logTransformedImage)),np.linalg.norm(logTransformedImage))
        return np.uint8(logTransformedImage)
    
   
   # Frequency Domain (DFT)
    def FrequencyDomainTransform(self, template):
        template = np.abs(np.fft.fft2(template))
        template = np.fft.fftshift(template)
        return template
    
    # Affine preprocessing 
    def Affine(self, template):
        a = -180 / 16
        b = 180 / 16
        r = a + (b - a) * np.random.uniform()
        rotationMatrix = cv2.getRotationMatrix2D((template.shape[1]/2, template.shape[0]/2), r, 1)
        rotatedTemplate = cv2.warpAffine(np.uint8(template * 255), rotationMatrix, (template.shape[1], template.shape[0]))
        rotatedTemplate = rotatedTemplate.astype(np.float32) / 255
        return rotatedTemplate
    #....................................................................
    
    '''
    # Find A gaussian peak image using the input image and the template where 
    # the target object is centered in the image.
    Reference github which helped us to implement this function
    '''
    def findSynthaticGaussianPeak(self, inputImage, template, segma = 100):
        # return a gaussian peak image. using template matching between
        # the template and the input image by calculating the distance between each pixel

        # find center of template (GT) w.r.t the input image
        x_c = template[0] + (template[2] / 2)
        y_c = template[1] + (template[3] / 2)
      
        
        width, height = inputImage.shape
        
        # to vectorise the operation we use mesh grid.
        width = np.arange(width) # give us a numpy array from 1 -> width
        height = np.arange(height) # give us a numpy array from 1 -> height
        array1, array2 = np.meshgrid(width, height) # array1: array of row index values, array2: array of column index values
        
        # Gaussian image
        gaussianArray = np.exp(-((np.square(array1 - x_c) + np.square(array2 - y_c)) / (2 * segma))) 
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
        return template,templateFreq, g, G, bBox_Pose
    
    #.......................................................................

    def InitializeMOSSE(self, template, G, iteration = 128,
                        learning_Rate= 0.125 , initFlag = 0 ):
        '''
        Parameters
        ----------
        template : Log of Cropped Image. 
        
        G : Gaussian Peak image in Fourier space (GT).
        
        iteration: number of iterations to train the filter.
        
        learning rate: allows the filter to quickly adapt to appearance
        changes while still maintaining a robust filter (from the paper)
        
        initFlag: a flag value to know if we are in the first frame or not.

        Returns
        -------
        Ai, Bi : MOSSEE Filter parameter values 

        '''
        # Resize the G (peak image) to be the same as templateFreq
        G = cv2.resize(G, dsize=(template.shape[1],template.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        ''' change preprocessing methods here'''
        #templateFreq = self.FrequencyDomainTransform(random_warp(template))
        template = self.logTransformTemplate(self.Affine(template))
        templateFreq = self.FrequencyDomainTransform(template)
        # if we are in the first frame we initilize the MOSSE filter.
        # Applying the H filter equations presented in the paper
        Ai = G * np.conjugate(templateFreq)
        Bi = (templateFreq * np.conjugate(templateFreq))
        # For the first frame we iterate to initialize the Filter H
        if initFlag == 0:  
            for i in range(iteration):
                Ai = Ai + (G * np.conjugate(templateFreq))
                Bi = Bi + (templateFreq * np.conjugate(templateFreq))
            # initial values of the Filter parameters     
            Ai = learning_Rate * Ai
            Bi = learning_Rate * Bi
        else: 
            # From the second frame till the end, Apply the parameters equations with the learning rate for each frame
            Ai = ((1-learning_Rate) * Ai) + (learning_Rate)*(G * np.conjugate(templateFreq))
            Bi = ((1-learning_Rate) * Bi) + (learning_Rate)*(templateFreq * np.conjugate(templateFreq))
            
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
                self.Ai,self.Bi = self.InitializeMOSSE(template, G)
                i = 1
            else: 
                # Regulizer
                reg = 0.01
                # Filter which will be updated in each frame 
                Hi = self.Ai/(self.Bi + reg)
                              
                # Draw a rectangle bounding box in the current frame which 
                # follows the moving target(by updating the tracking window)
                curr_template = curr_frame_gray[self.trackingWindow[1]:self.trackingWindow[3],
                                    self.trackingWindow[0]:self.trackingWindow[2]]

                curr_template = cv2.resize(curr_template, (self.bBox_Pose[2], self.bBox_Pose[3]))
                
                ''' change preprocessing methods here'''
                curr_template = self.logTransformTemplate(curr_template)
                #curr_template = pre_process(random_warp(curr_template))

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
                curr_template = cv2.resize(curr_template, (self.bBox_Pose[2], self.bBox_Pose[3]))
               
                ''' change preprocessing methods here'''
                curr_template = self.logTransformTemplate(curr_template)
                #curr_template = pre_process(random_warp(curr_template))

                curr_templateF = self.FrequencyDomainTransform(curr_template)

                # Update A and B
                self.Ai,self.Bi = self.InitializeMOSSE(curr_templateF, G, iteration=0,initFlag=1)


            # Visulize the tracking window 
            cv2.rectangle(curr_frame, (self.trackingWindow[0], self.trackingWindow[1]),
                          (self.trackingWindow[2], self.trackingWindow[3]), (255, 0, 0), 2)
            cv2.imshow('test', curr_frame)
            cv2.waitKey(100)
            # counter 
            i = i + 1 
            

    def trackFrames(self, path): 

       # initialize  trcking window position.

       # counter for the video frames
       i = 0
       # list of frames 
       listOfFrames = self.readFrames(path)
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
               self.Ai,self.Bi = self.InitializeMOSSE(template, G)
              
           else: 
               # Regulizer
               reg = 0.01
               # Filter which will be updated in each frame 
               Hi = self.Ai/(self.Bi + reg)
                             
               # Draw a rectangle bounding box in the current frame which 
               # follows the moving target(by updating the tracking window)
               curr_template = curr_frame_gray[self.trackingWindow[1]:self.trackingWindow[3],
                                   self.trackingWindow[0]:self.trackingWindow[2]]


               curr_template = cv2.resize(curr_template, (self.bBox_Pose[2], self.bBox_Pose[3]))
               curr_template = self.logTransformTemplate(curr_template)
               # Currrent template in Fourier Space
               curr_templateF = self.FrequencyDomainTransform(curr_template)
               
               # The multiplication of the updated Filter H conjugate and curr_templateF
               # results in a peak image (G) which indicates the new location
               # of the target object.we are going to call it newObjLoction_G
               newObjLoction_G = Hi * curr_templateF
               #newObjLoction_G = np.uint8(np.fft.ifft2(newObjLoction_G))
                       
               # find the position where the peak is, the center(where we will find the target object)
               #newObjLoction_G = self.FrequencyDomainTransform(newObjLoction_G)
               max_value = np.max(newObjLoction_G)
               max_pos = np.where(newObjLoction_G == max_value)

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
               curr_template = cv2.resize(curr_template, (self.bBox_Pose[2], self.bBox_Pose[3]))
               curr_template = self.logTransformTemplate(curr_template)
               curr_templateF = self.FrequencyDomainTransform(curr_template)
               # online update...
               self.Ai,self.Bi = self.InitializeMOSSE(curr_templateF, G, iteration=0,initFlag=1)


           # Visulize the tracking window 
           cv2.rectangle(curr_frame, (self.trackingWindow[0], self.trackingWindow[1]),
                      (self.trackingWindow[2], self.trackingWindow[3]), (255, 0, 0), 2)
           cv2.imshow('test', curr_frame)
           cv2.waitKey(100)
           # counter 
           i = i + 1 
        
        
            
            

#................................Test.......................................              
start = MOSSE()  
# test data 1 with images
# your data path    
path = './Data/test5'      
start.trackFrames(path)

# test data video 
# your video path 
#path = './Data/testVideo.avi'
#start.trackVideo(path)
