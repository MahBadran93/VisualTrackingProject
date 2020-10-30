import numpy as np 
import matplotlib.pyplot as plt 
import cv2

im1 = cv2.imread('download.png')
im = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

class PreProcess:
    '''
        preprocess operations applied on a multiple frames on 
        video. 
        
    '''
    def __init__(self):
        self.template = [] 
    
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
    
    def logTransformTemplate(self, template):
        const = 255 / np.log(1 + np.max(template)) 
        logTransformedImage = const * (np.log(template + 1)) 
        #logTransformedImage = (logTransformedImage - np.mean(logTransformedImage)) / np.linalg.norm(logTransformedImage)
        return np.uint8(logTransformedImage)
   
    def FrequencyDomainTransform(self, template):
        template = np.abs(np.fft.fft2(template))
        template = np.fft.fftshift(template)
        return template
    
    def findSynthaticGaussianPeak(self, inputImage, template, segma = 2):
        # return a gaussian peak image. calculate distance between each pixel
        # with the center of our target object (template(ground truth))
        
        # find center of template (GT)
        x_c = (template[0]  +template[2]) / 2
        y_c = (template[1] + template[3]) / 2
        
        width, height = inputImage.shape
        
        # to vectorise the operation we use mesh grid.
        width = np.arange(width) # give us a numpy array from 1 -> width
        height = np.arange(height) # give us a numpy array from 1 -> height
        array1, array2 = np.meshgrid(width, height) # array1: row indices values, array2: column indices values
        
        gaussianArray = np.exp(-((np.square(array1 - x_c) + np.square(array2 - y_c)) / (2 * segma))) 
        
        return gaussianArray
         
    
        
    
        
inits = PreProcess()
template = inits.CreateTemplate(im)


logTemplate = inits.logTransformTemplate(template)
#plt.imshow(logTemplate,cmap='gray')
#plt.show()
#plt.imshow(inits.FrequencyDomainTransform(logTemplate),cmap='gray')
#plt.show()
#print(np.max(logTemplate),np.min(logTemplate))
#cv2.imshow('cropedImg', np.fft.fftshift(np.abs(np.fft.fft2(logTemplate)))) 
#cv2.waitKey(0)
#cv2.destroyAllWindows()