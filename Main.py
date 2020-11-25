import numpy as np 
import matplotlib.pyplot as plt 
import cv2



class MOSSE:
    '''
        - Preprocessing step is performed on every frame of the video before
        initialization or tracking(before using MOSSE Filter).
        
    '''
    def __init__(self):
        self.template = [] 
        
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
    #....................................................................
    
    '''
    # Find A gaussian peak image using the input image and the template where 
    # the target object is centered in the image.
    Reference github which helped us to implement this function
    '''
    def findSynthaticGaussianPeak(self, inputImage, template, segma = 30):
        # return a gaussian peak image. using template matching between
        # the template and the input image by calculating the distance between each pixel

        # find center of template (GT)
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
  
    def InitializeMOSSE(self, templateFreq, G, iteration = 128,
                        learning_Rate= 0.125 , initFlag = 0 ):
        '''
        Parameters
        ----------
        templateFreq : Cropped Image in Fourier space. 
        
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
        G = cv2.resize(G, dsize=(templateFreq.shape[1],templateFreq.shape[0]), interpolation=cv2.INTER_CUBIC)
        
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
    
        
    def track(self): 

        # initialize  trcking window position.
       
        # get input video 
        vcap = cv2.VideoCapture('./Data/testVideo.avi')
        
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
                # extract template, gaussian peak.
                # In time(template,g) and frequency domain(templateF,G)
                template,templateF,g,G, bBox_Pose = self.ceateDataSet(curr_frame_gray)
                # current frame tracking window coordinates
                bBox_Pose = np.array(bBox_Pose).astype(np.int64)
                trackingWindow = np.array([bBox_Pose[0],
                                  bBox_Pose[1],
                                  bBox_Pose[0]+bBox_Pose[2],
                                  bBox_Pose[1]+bBox_Pose[3]]).astype(np.int64)
        
                #initialize the filter(using only the first frame).
                # Eq5 in the paper.
                Ai,Bi = self.InitializeMOSSE(templateF, G)
                i = 1
            else: 
                # Regulizer
                reg = 0
                # Filter which will be updated in each frame 
                Hi = Ai/(Bi + reg)
                              
                # Draw a rectangle bounding box in the current frame which 
                # follows the moving target(by updating the tracking window)
                curr_template = curr_frame_gray[trackingWindow[1]:trackingWindow[3],
                                    trackingWindow[0]:trackingWindow[2]]

                #curr_template = cv2.resize(fi, (bBox_Pose[2], bBox_Pose[3]))
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

                changeIn_Y = int(np.mean(max_pos[0]) - newObjLoction_G.shape[0] / 2)
                changeIn_X = int(np.mean(max_pos[1]) - newObjLoction_G.shape[1] / 2)
                
                # update the position...
                bBox_Pose[0] = bBox_Pose[0] + changeIn_X
                bBox_Pose[1] = bBox_Pose[1] + changeIn_Y

                # trying to get the clipped position [xmin, ymin, xmax, ymax]
                trackingWindow[0] = bBox_Pose[0] 
                trackingWindow[1] = bBox_Pose[1] 
                trackingWindow[2] = bBox_Pose[0]+bBox_Pose[2] 
                trackingWindow[3] = bBox_Pose[1]+bBox_Pose[3] 
                trackingWindow = trackingWindow.astype(np.int64)

                # get the current template using the updated tracking winow..
                curr_template = curr_frame_gray[trackingWindow[1]:trackingWindow[3], trackingWindow[0]:trackingWindow[2]]
                curr_template = cv2.resize(curr_template, (bBox_Pose[2], bBox_Pose[3]))
                curr_template = self.logTransformTemplate(curr_template)
                curr_templateF = self.FrequencyDomainTransform(curr_template)
                #plt.imshow(Fi)
                #plt.show()
                # online update...
                Ai,Bi = self.InitializeMOSSE(curr_templateF, G, iteration=0,initFlag=1)


            # Visulize the tracking window 
            cv2.rectangle(curr_frame, (trackingWindow[0], trackingWindow[1]),
                          (trackingWindow[2], trackingWindow[3]), (255, 0, 0), 2)
            cv2.imshow('test', curr_frame)
            cv2.waitKey(100)
            # counter 
            i = i + 1 

            
        
            
            

              
start = MOSSE()            
start.track()
    



'''         
vcap = cv2.VideoCapture('./Data/testVideo.avi')
while(True):
    ret, frame = vcap.read()
    #print cap.isOpened(), ret
    if frame is not None:
        # Display the resulting frame
        cv2.imshow('frame',frame)
        # Press q to close the video windows before it ends if you want
        if cv2.waitKey(22) & 0xFF == ord('q'):
            break
    else:
        print ("Frame is None")
        break
'''            
