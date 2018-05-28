// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// JeVois Smart Embedded Machine Vision Toolkit - Copyright (C) 2016 by Laurent Itti, the University of Southern
// California (USC), and iLab at USC. See http://iLab.usc.edu and http://jevois.org for information about this project.
//
// This file is part of the JeVois Smart Embedded Machine Vision Toolkit.  This program is free software; you can
// redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software
// Foundation, version 2.  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
// License for more details.  You should have received a copy of the GNU General Public License along with this program;
// if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
//
// Contact information: Laurent Itti - 3641 Watt Way, HNB-07A - Los Angeles, CA 90089-2520 - USA.
// Tel: +1 213 740 3527 - itti@pollux.usc.edu - http://iLab.usc.edu - http://jevois.org
// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*! \file */
#include <jevois/Core/Module.H>
#include <jevois/Image/RawImageOps.H>


#include <iostream>
#include<sstream>
#include <atomic>        

#include<opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/imgproc.hpp>
#include "opencv2/objdetect.hpp"
#include <opencv2/ml.hpp>
#include "opencv2/features2d.hpp"

#include <string>
#include <vector>

using namespace cv::ml;
using namespace cv;
using namespace std;


//Ptr<SVM> svm = Algorithm::load<SVM>("7.xml");

// global variables ///////////////////////////////////////////////////////////////////////////////
const int MIN_CONTOUR_AREA = 500;   //possible changes for training data with small pixels
// const int MIN_CONTOUR_AREA1 = 40;
// const float MIN_CONTOUR_AREA2 = 27.5;
const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 20;

int SZ = 20;
float affineFlags = WARP_INVERSE_MAP|INTER_LINEAR;

    vector<Mat> testCells;

    vector<int> testLabels;
    vector<int> testLabels1;

    

    vector<Mat> deskewedTestCells;
    vector<Mat> deskewedTestCells1;



    std::vector<std::vector<float> > testHOG;
    std::vector<std::vector<float> > testHOG1;

     Mat testResponse;
    Mat testResponse1;
   Mat matTestingNumbers;
char str;
Ptr<SVM>  svm1;
   Ptr<SVM>  svm;


   
    class HelloJeVois : public jevois::Module
 {
   public:
  
  HelloJeVois(std::string const & instance) : jevois::Module(instance) {  }
    class ContourWithData {
    public:
     
     std::vector<cv::Point> ptContour;    
     cv::Rect boundingRect;       
     float fltArea;       
 

 
    ///////////////////////////////////////////////////////////////////////////////////////////////
    static bool sortByBoundingRectXPosition(const ContourWithData& cwdLeft, const ContourWithData& cwdRight) {      // this function allows us to sort
        return(cwdLeft.boundingRect.x < cwdRight.boundingRect.x);                                                   // the contours from left to right???
    }

}; 
   

    virtual ~HelloJeVois() { }
    //! Processing function

 void postInit() override
   {
       b.store(true);
 itsRunFut = std::async(std::launch::async, &HelloJeVois::run,);
  }
 
 void postUninit() override
     {
       // Signal end of run:
       b.store(false);
    
  }

 Ptr<SVM>  svm1 = Algorithm::load<SVM>("/jevois/modules/Tutorial/HelloJeVois/4.xml");
 

    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {   //  cv::Mat matTestingNumbers = cv::imread("screenshot1.png");            // read in the test numbers image
        std::atomic<bool> b (false);
         
HOGDescriptor hog(
        Size(20,20), //winSize
        Size(8,8), //blocksize
        Size(4,4), //blockStride,
        Size(8,8), //cellSize,
                 9, //nbins,
                  1, //derivAper,
                 -1, //winSigma,
                  0, //histogramNormType,
                0.2, //L2HysThresh,
                  0,//gammal correction,
                  64,//nlevels=64
                  1);
   std::vector<HelloJeVois::ContourWithData> allContoursWithData;           // declare empty vectors,
   std::vector<HelloJeVois::ContourWithData> validContoursWithData;
   cv::Mat matBlurred;             // declare more image variables
   cv::Mat matThresh;              //
    cv::Mat matThreshCopy;
  

     jevois::RawImage const inimg = inframe.get();

     
      inimg.require("input", inimg.width, inimg.height, V4L2_PIX_FMT_YUYV);
      
      // Wait for an image from our gadget driver into which we will put our results:
      jevois::RawImage outimg;

      cv::Mat grayImage = jevois::rawimage::convertToCvGray(inimg);

         std::future<void> fut = std::async(std::launch::async, [&]() {
          outimg = outframe.get();
          // Enforce that the input and output formats and image sizes match:
          outimg.require("output", inimg.width, inimg.height, inimg.fmt);
         
     
        cv::GaussianBlur(grayImage,matBlurred,cv::Size(5, 5), 0);        // output image
        cv::adaptiveThreshold(matBlurred,                           // input image
        matThresh,                            // output image
        255,                                  // make pixels that pass the threshold full white
        cv::ADAPTIVE_THRESH_GAUSSIAN_C,       // use gaussian rather than mean, seems to give better results
        cv::THRESH_BINARY_INV,                // invert so foreground will be white, background will be black
        11,                                   // size of a pixel neighborhood used to calculate threshold value
        2);  
                  // smoothing window width and height in pixels
       // matThreshCopy = matThresh.clone();  
        std::vector<std::vector<cv::Point> > ptContours;   
        std::vector<cv::Vec4i> v4iHierarchy; 

        cv::findContours(matThresh,             // input image, make sure to use a copy since the function will modify this image in the course of finding contours
        ptContours,                             // output contours
        v4iHierarchy,                           // output hierarchy
        cv::RETR_EXTERNAL,                      // retrieve the outermost contours only
        cv::CHAIN_APPROX_SIMPLE); 

        for (int i = 0; i < ptContours.size(); i++) {               // for each contour
        HelloJeVois::ContourWithData contourWithData;                                                    // instantiate a contour with data object
        contourWithData.ptContour = ptContours[i];                                          // assign contour to contour with data
       contourWithData.boundingRect = cv::boundingRect(contourWithData.ptContour);         // get the bounding rect
        contourWithData.fltArea = cv::contourArea(contourWithData.ptContour);               // calculate the contour area
         if (contourWithData.fltArea>500)
       { 
        validContoursWithData.emplace_back(contourWithData);      
             // add contour with data object to list of all contours with data
       }
    }
   
     std::sort(validContoursWithData.begin(), validContoursWithData.end(), HelloJeVois::ContourWithData::sortByBoundingRectXPosition);


    for (int i = 0; i < validContoursWithData.size(); i++) {   
     cv::rectangle(grayImage,                            // draw rectangle on original image
           validContoursWithData[i].boundingRect,        // rect to draw
            cv::Scalar(0, 255, 0),                        // green
            2);   
     cv::Mat matROI = matThresh(validContoursWithData[i].boundingRect); 
                                                // thickness
        
              // get ROI image of bounding rect
      cv::Mat matROIResized;
      cv::resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));
     //testCells.emplace_back(matROIResized);
     vector<float> descriptors;
      hog.compute(matROIResized,descriptors);
      testHOG1.emplace_back(descriptors); }
 

    int descriptor_size1 = 144;
    Mat testMat1(testHOG1.size(),descriptor_size1,CV_32FC1);
        Mat testResponse1;

  for(int i = 0;i<testHOG1.size();i++){
        for(int j = 0;j<descriptor_size1;j++){
            testMat1.at<float>(i,j) = testHOG1[i][j]; 
        }
    }
 
  Ptr<SVM>  svm1 = Algorithm::load<SVM>("/jevois/modules/Tutorial/HelloJeVois/4.xml");

  // string x;
  // char s[testResponse1.rows];
  svm1->predict(testMat1, testResponse1);


//float   matAsString = 11;
jevois::rawimage::pasteGreyToYUYV(grayImage, outimg, 0, 0);


for(int i = 0; i < testResponse1.rows; i++)
{  int t;
    t=50*i;
float   matAsString = testResponse1.at<float>(i,0);
jevois::rawimage::writeText(outimg,  std::to_string(matAsString) ,
                             10, t, jevois::yuyv::LightGreen,jevois::rawimage::Font20x38);
    }    
 

    });           


  

      fut.get();
     

      inframe.done(); // NOTE: optional here, inframe destructor would call it anyway

      outframe.send();
      // Send the outp outframe.send(); // NOTE: optional here, outframe destructor would call it anyway
   }

protected:
 void run()
 { 
    while (b.load())
  {
      Ptr<SVM>  svm1 = Algorithm::load<SVM>("/jevois/modules/Tutorial/HelloJeVois/4.xml"); }
       }
      std::future<void> itsRunFut;
   std::atomic<bool> b; //!< Future for our run() thread   
//  Ptr<SVM> svm1 = Algorithm::load<SVM>("8.xml");
//  protected:
};
JEVOIS_REGISTER_MODULE(HelloJeVois);

