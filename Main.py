import numpy as np 
import matplotlib.pyplot as plt 
import cv2

im1 = cv2.imread('download.png')
im = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)



class PreProcess:
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
        bBox = cv2.selectROI('BoundaryBox',
                                     inputImage,
                                     showCrosshair = True
                                     )
        
        self.template = inputImage[int(bBox[1]):int(bBox[1]+bBox[3]),
                              int(bBox[0]):int(bBox[0]+bBox[2])]
        
        self.template = np.array(self.template).astype(np.int64)
        return self.template
    
    # return the 4 point poistions of the croped template w.r.t the input image
    def getCropedTemplate4Points(self, inputImage): 
        bBox = cv2.selectROI('BoundaryBox',
                                     inputImage,
                                     showCrosshair = True
                                     )
        return bBox
    
    # Log transformation 
    def logTransformTemplate(self, template):
        const = 255 / np.log(1 + np.max(template)) 
        logTransformedImage = const * (np.log(template + 1)) 
        #logTransformedImage = (logTransformedImage - np.mean(logTransformedImage)) / np.linalg.norm(logTransformedImage)
        return np.uint8(logTransformedImage)
   # Frequency Domain (DFT)
    def FrequencyDomainTransform(self, template):
        template = np.abs(np.fft.fft2(template))
        template = np.fft.fftshift(template)
        return template
    
    #.........................................................................
    
    
    '''
    # Find A gaussian peak image using the input image and the template where 
    # the target object is centered in the image.
    '''
    def findSynthaticGaussianPeak(self, inputImage, template, segma = 2):
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
        print(gaussianArray.shape)
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
        template = self.logTransformTemplate(self.CreateTemplate(frame))
        # input template frequency domain 
        templateFreq = self.FrequencyDomainTransform(template) 
        # gaussian peak centered template
        g = self.findSynthaticGaussianPeak(frame, self.getCropedTemplate4Points(frame))
        print(template.shape, templateFreq.shape)
        # frequency domain gaussian template
        G = self.FrequencyDomainTransform(g)
        return template,templateFreq, g, G 
    
    #.......................................................................
    
    '''
    # Apply Affine Transformation on the cropped image(the template)
    # This function can be used during the initialzation process
    '''
    def AFFineTransformation(self, template):
        a = -180 / 16
        b = 180 / 16
        r = a + (b - a) * np.random.uniform()
        matrix_rot = cv2.getRotationMatrix2D((template.shape[1]/2, template.shape[0]/2), r, 1)
        img_rot = cv2.warpAffine(np.uint8(template * 255), matrix_rot, (template.shape[1], template.shape[0]))
        img_rot = img_rot.astype(np.float32) / 255
        return img_rot
        
    # as the paper implies, Mosse filter needs training images (templates) in frequency domain, 
    # and G(ground truth), a croped image of the object to track (gaussian peak centered) in frequency domain.
    # we use frequency domain images to simplify the calculations (use element wise mult. instead of convolution).
    # in tracking, for each frame we will call MOSSE filter 
    def InitializeMOSSE(self, templateFreq, G, iteration = 100,
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
        G = cv2.resize(G, dsize=(templateFreq.shape[1],templateFreq.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        # if we are in the first frame we initilize the MOSSE filter.
        # Applying the H filter equations presented in the paper
        Ai = G * np.conjugate(templateFreq)
        Bi = templateFreq * np.conjugate(templateFreq)
        if initFlag == 0:
            for i in range(iteration):
                Ai = Ai + G * np.conjugate(templateFreq)
                Bi = Bi + templateFreq * np.conjugate(templateFreq)
        else: 
            Ai = (learning_Rate * Ai) + (1-learning_Rate)*(G * np.conjugate(templateFreq))
            Bi = (learning_Rate * Bi) + (1-learning_Rate)*(templateFreq * np.conjugate(templateFreq))
            
        return Ai, Bi 
    
        
    def track(self): 
        
        #..............get the input data and initilize some param...........
        
        # Video or list of frames. Our input.
        framelist = []
        
        # initialize  trcking window position.
        # in the first frame it will be in the same position as the template
        trackingWindow = np.array([self.template[0],
                               self.template[1],
                               self.template[0]+self.template[2],
                               self.template[1]+self.template[3]])
        

        for i in range(len(framelist)):
            curr_frame = cv2.imread(framelist[i])
            curr_frame_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
            template,templateF,g,G = inits.ceateDataSet(framelist[i])

            # in the first frame, the tracking window will be in the same position
            # as the croped template. 
            # also we initialize the filter in the first frame.
            if i == 0: 
                # initialize the filter. eq5 in the paper.
                Ai,Bi = self.InitializeMOSSE(templateF, G)
                # Draw a rectangle bounding box in the current frame which 
                # follows the moving target.
                cv2.rectangle(framelist[i], (trackingWindow[0], trackingWindow[1]),
                              (trackingWindow[3], trackingWindow[4]),
                              (0, 255, 0), 2)
            else: 
                Ai,Bi = self.InitializeMOSSE(templateF, G, iteration=0,initFlag=1)
                
            
            # Visulize the tracking window 
            cv2.rectangle(curr_frame, (trackingWindow[0], trackingWindow[1]),
                          (trackingWindow[3], trackingWindow[4]), (0, 255, 0), 2)
            cv2.imshow('demo', curr_frame)
            cv2.waitKey(100)    
            
    



         
    
#................................................
        

inits = PreProcess()
initalize = inits.InitializeMOSSE(templateF,G)

print('template', template.shape,'inits', initalize.shape)
aff = inits.AFFineTransformation(template)
#logTemplate = inits.logTransformTemplate(template)
plt.imshow(initalize ,cmap='gray')
plt.show()
plt.imshow(g,cmap='gray')

#plt.imshow(inits.FrequencyDomainTransform(logTemplate),cmap='gray')
#plt.show()
#print(np.max(logTemplate),np.min(logTemplate))
#cv2.imshow('cropedImg', np.fft.fftshift(np.abs(np.fft.fft2(logTemplate)))) 
#cv2.waitKey(0)
#cv2.destroyAllWindows()

