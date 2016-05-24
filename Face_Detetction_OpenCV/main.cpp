//
//  main.cpp
//  Face_Detetction_OpenCV
//
//  Created by Katsuya Fujii on 5/24/16.
//  Copyright © 2016 Katsuya Fujii. All rights reserved.
//

#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/ml.h>


void drawSquare(IplImage* in, int x, int y,
              int width, int height, int size);



int main (int argc, char **argv)
{   double w = 320, h = 240;
    int i, c;
    IplImage *src_img = 0, *src_gray = 0;
    const char *cascade_name = "/Users/kty0515/Documents/Xcode/Face_Detetction_OpenCV/Face_Detetction_OpenCV/haarcascade_frontalface_alt.xml";
    CvHaarClassifierCascade *cascade = 0;
    CvMemStorage *storage = 0;
    CvSeq *faces;
    
    cascade = (CvHaarClassifierCascade *) cvLoad (cascade_name, 0, 0, 0);
  
    

    CvCapture *capture = cvCreateCameraCapture(0);
    cvSetCaptureProperty (capture, CV_CAP_PROP_FRAME_WIDTH, w);
    cvSetCaptureProperty (capture, CV_CAP_PROP_FRAME_HEIGHT, h);
        cvNamedWindow ("Capture", CV_WINDOW_AUTOSIZE);
    assert(capture != NULL);
    
    while (1) {
        src_img = cvQueryFrame (capture);
        src_gray = cvCreateImage (cvGetSize(src_img), IPL_DEPTH_8U, 1);
        
        storage = cvCreateMemStorage (0);
        cvClearMemStorage (storage);
        cvCvtColor (src_img, src_gray, CV_BGR2GRAY);
        cvEqualizeHist (src_gray, src_gray);
        
        faces = cvHaarDetectObjects (src_gray, cascade, storage,
                                     1.11, 4, 0, cvSize (40, 40));
        for (i = 0; i < (faces ? faces->total : 0); i++) {
            CvRect *r = (CvRect *) cvGetSeqElem (faces, i);
            drawSquare(src_img, r->x, r->y, r->width, r->height, 20);
        }
        
        cvShowImage("Capture", src_img);
        cvReleaseImage(&src_gray);
        
        c = cvWaitKey (30);
        if (c == '\x1b')
            break;
    }
    
    cvReleaseCapture (&capture);
    cvDestroyWindow ("Capture");
    
    return 0;
}

void drawSquare(IplImage* in, int x0, int y0,
              int width, int height, int size)
{
    int b, g, r, col, row;
    
    int xMin = size*(int)floor((double)x0/size);
    int yMin = size*(int)floor((double)y0/size);
    int xMax = size*(int)ceil((double)(x0+width)/size);
    int yMax = size*(int)ceil((double)(y0+height)/size);
   
    cvRectangle(in, cvPoint(xMin,yMax), cvPoint(xMax, yMin), cvScalar(250,255,0), 0.1, 0.1);
    
}

