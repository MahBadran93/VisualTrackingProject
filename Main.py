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
            croped image of the object we want to track.

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
   
    def FrequencyDomainTransform(self, logedTemplate):
        logedTemplate = np.abs(np.fft.fft2(logedTemplate))
        logedTemplate = np.fft.fftshift(logedTemplate)
        return logedTemplate
    
        
inits = PreProcess()
template = inits.CreateTemplate(im)


logTemplate = inits.logTransformTemplate(template)
plt.imshow(logTemplate,cmap='gray')
plt.show()
plt.imshow(inits.FrequencyDomainTransform(logTemplate),cmap='gray')
plt.show()
#print(np.max(logTemplate),np.min(logTemplate))
#cv2.imshow('cropedImg', np.fft.fftshift(np.abs(np.fft.fft2(logTemplate)))) 
#cv2.waitKey(0)
#cv2.destroyAllWindows()