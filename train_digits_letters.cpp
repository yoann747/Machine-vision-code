#include <iostream>
#include<sstream>

#include<opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/objdetect.hpp"
#include <opencv2/ml.hpp>

#include <string>
#include <vector>


using namespace cv::ml;
using namespace cv;
using namespace std;


// global variables ///////////////////////////////////////////////////////////////////////////////
const int MIN_CONTOUR_AREA = 150;   //possible changes for training data with small pixels
const int MIN_CONTOUR_AREA1 = 40;
const float MIN_CONTOUR_AREA2 = 27.5;

const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 20;

static const char ini[4] =  { 'L', 'R', 'F', 'B', };
    

class ContourWithData {
public:
    std::vector<cv::Point> ptContour;    
     cv::Rect boundingRect;       
     float fltArea;       
    std::vector<cv::Point> ptContour1;         // contour
     cv::Rect boundingRect1;                 // bounding rect for contour
     float fltArea1;                       // area of contour  
     std::vector<cv::Point> ptContour2;         // contour
     cv::Rect boundingRect2;                 // bounding rect for contour
     float fltArea2;       

    // member variables ///////////////////////////////////////////////////////////////////////////
                    // area of contour
                                                ///////////////////////////////////////////////////////////////////////////////////////////////
    bool checkIfContourIsValid() {                              // obviously in a production grade program
        if (fltArea < MIN_CONTOUR_AREA) return false;           // we would have a much more robust function for 
        return true;                                            // identifying if a contour is valid !!
    }
 
    ///////////////////////////////////////////////////////////////////////////////////////////////
    static bool sortByBoundingRectXPosition(const ContourWithData& cwdLeft, const ContourWithData& cwdRight) {      // this function allows us to sort
        return(cwdLeft.boundingRect.x < cwdRight.boundingRect.x);                                                   // the contours from left to right???
    }

                                                //////////////////////////////////////////////////////////////////////////////////////////////
    bool checkIfContourIsValid1() {                              // obviously in a production grade program
        if (fltArea1 < MIN_CONTOUR_AREA1) return false;           // we would have a much more robust function for 
        return true;                                            // identifying if a contour is valid !!
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////
    static bool sortByBoundingRectXPosition1(const ContourWithData & cwdLeft, const ContourWithData& cwdRight) {      // this function allows us to sort
        return(cwdLeft.boundingRect1.y < cwdRight.boundingRect1.y);                                                   // the contours from left to right???
    }


    // member variables ///////////////////////////////////////////////////////////////////////////


                                                ///////////////////////////////////////////////////////////////////////////////////////////////
   
    bool checkIfContourIsValid2() {                              // obviously in a production grade program
        if (fltArea2 < MIN_CONTOUR_AREA2) return false;           // we would have a much more robust function for 
        return true;                                            // identifying if a contour is valid !!
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////
    static bool sortByBoundingRectXPosition2(const ContourWithData & cwdLeft, const ContourWithData& cwdRight) {      // this function allows us to sort
        return(cwdLeft.boundingRect2.y < cwdRight.boundingRect2.y);                                                   // the contours from left to right???
    }
};



std::string strFinalString;  

string pathName = "digits.png";
string pathName1 = "letter2.png";
//string pathName2 = "letter1-lossless.png";


      // show threshold image for reference

int SZ = 20;
float affineFlags = WARP_INVERSE_MAP|INTER_LINEAR;

Mat deskew(Mat& img){
    Moments m = moments(img);
    if(abs(m.mu02) < 1e-2){
        return img.clone();
    }
    float skew = m.mu11/m.mu02;
    Mat warpMat = (Mat_<float>(2,3) << 1, skew, -0.5*SZ*skew, 0, 1, 0);
    Mat imgOut = Mat::zeros(img.rows, img.cols, img.type());
    warpAffine(img, imgOut, warpMat, imgOut.size(),affineFlags);

    return imgOut;
} 



void loadTrainTestLabel(string &pathName, vector<Mat> &trainCells,vector<Mat> &trainCells1, vector<Mat> &testCells,vector<Mat> &trialcells,vector<int> &trainLabels,vector<int> &trainLabels1, vector<int> &testLabels)
{
    std::vector<ContourWithData> allContoursWithData;           // declare empty vectors,
    std::vector<ContourWithData> validContoursWithData;         // we will fill these shortly
    std::vector<ContourWithData> allContoursWithData1;           // declare empty vectors,
    std::vector<ContourWithData> validContoursWithData1;         // we will fill these shortly
    std::vector<ContourWithData> allContoursWithData2;           // declare empty vectors,
    std::vector<ContourWithData> validContoursWithData2;         // we will fill these shortly

    Mat img = imread(pathName,0);
   // Mat img1 = imread(pathName1,CV_LOAD_IMAGE_GRAYSCALE);
   
    int ImgCount = 0;
    for(int i = 0;i < img.rows ; i = i + SZ)
    {
        cv::Mat matROI3; 
        cv::Mat matROI3Resized; 
        for(int j = 0; j < img.cols; j = j + SZ)
        {
            Mat digitImg = (img.colRange(j,j+SZ).rowRange(i,i+SZ)).clone();

            if(j < int(img.cols))
            { 
             trainCells.push_back(digitImg);
  
            ImgCount++;

        }
    }

for(int i = 0;i < img.rows ; i = i + 5*SZ)
    {
        cv::Mat matROI2; 
        cv::Mat matROI2Resized;


        for(int j = 0; j < 1 ; j = j + SZ)
        {
            Mat digitImg2 = (img.colRange(j,j+SZ).rowRange(i,i+SZ)).clone();
            if(j < int(img.cols))
            {
            matROI2 = digitImg2(Rect(4,2,12,16));
            cv::resize(matROI2, matROI2Resized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT)); 

              trialcells.push_back(matROI2Resized);

            }   
            }
            }

  
  cv::Mat matTestingNumbers = cv::imread("test05.png");            // read in the test numbers image
  cv::Mat imgTrainingletters = cv::imread("letter2.png");
  cv::Mat imgTrainingnumbers = cv::imread("digits.png");



  if (matTestingNumbers.empty()) {                                // if unable to open image
        std::cout << "error: image not read from file\n\n";         // show error message on command line
        //return 0;                                                  // and exit program
    }
    
 

    cv::Mat matGrayscale;           //
    cv::Mat matBlurred;             // declare more image variables
    cv::Mat matThresh;              //
    cv::Mat matThreshCopy;          //

    cv::Mat imgGrayscale;               // 
    cv::Mat imgBlurred;                 // declare various images
    cv::Mat imgThresh;                  //
    cv::Mat imgThreshCopy;              //

    cv::Mat imgGrayscale1;               // 
    cv::Mat imgBlurred1;                 // declare various images
    cv::Mat imgThresh1;                  //
    cv::Mat imgThreshCopy1;              //


    cv::Mat imgTrainingnumbers1;
    bitwise_not ( imgTrainingnumbers, imgTrainingnumbers1 );

    cv::cvtColor(matTestingNumbers, matGrayscale, cv::COLOR_BGR2GRAY);         // convert to grayscale
    cv::cvtColor(imgTrainingletters, imgGrayscale, cv::COLOR_BGR2GRAY);         // convert to grayscale
    cv::cvtColor(imgTrainingnumbers1, imgGrayscale1, cv::COLOR_BGR2GRAY);         // convert to grayscale


        // blur
    cv::GaussianBlur(matGrayscale,              // input image
        matBlurred,                // output image
        cv::Size(5, 5),            // smoothing window width and height in pixels
        0);                        // sigma value, determines how much the image will be blurred, zero makes function choose the sigma value
    cv::GaussianBlur(imgGrayscale,              // Blurs an image using a Gaussian filter.  input image
        imgBlurred,                             // output image
        cv::Size(5, 5),                         // smoothing window width and height in pixels
        0);                                     // sigma value, determines how much the image will be blurred, zero makes function choose the sigma value
    cv::GaussianBlur(imgGrayscale1,              // Blurs an image using a Gaussian filter.  input image
        imgBlurred1,                             // output image
        cv::Size(5, 5),                         // smoothing window width and height in pixels
        0);                                     // sigma value, determines how much the image will be blurred, zero makes function choose the sigma value
    
      
    
      
    cv::adaptiveThreshold(matBlurred,                           // input image
        matThresh,                            // output image
        255,                                  // make pixels that pass the threshold full white
        cv::ADAPTIVE_THRESH_GAUSSIAN_C,       // use gaussian rather than mean, seems to give better results
        cv::THRESH_BINARY_INV,                // invert so foreground will be white, background will be black
        11,                                   // size of a pixel neighborhood used to calculate threshold value
        2);     
                                      // constant subtracted from the mean or weighted mean
    cv::adaptiveThreshold(imgBlurred,           // input image
        imgThresh,                              // output image
        255,                                    // make pixels that pass the threshold full white
        cv::ADAPTIVE_THRESH_GAUSSIAN_C,         // use gaussian rather than mean, seems to give better results
        cv::THRESH_BINARY_INV,                  // invert so foreground will be white, background will be black
        11,                                     // size of a pixel neighborhood used to calculate threshold value
        2);                                     // constant subtracted from the mean or weighted mean
    
     cv::adaptiveThreshold(imgBlurred1,           // input image
        imgThresh1,                              // output image
        255,                                    // make pixels that pass the threshold full white
        cv::ADAPTIVE_THRESH_GAUSSIAN_C,         // use gaussian rather than mean, seems to give better results
        cv::THRESH_BINARY_INV,                  // invert so foreground will be white, background will be black
        11,                                     // size of a pixel neighborhood used to calculate threshold value
        2);                                     // constant subtracted from the mean or weighted mean
    

    matThreshCopy = matThresh.clone();  
    imgThreshCopy = imgThresh.clone();          // make a copy of the thresh image, this in necessary b/c findContours modifies the
    imgThreshCopy1 = imgThresh1.clone();          // make a copy of the thresh image, this in necessary b/c findContours modifies the

 


    std::vector<std::vector<cv::Point> > ptContours;        // declare a vector for the contours
    std::vector<cv::Vec4i> v4iHierarchy;                    // declare a vector for the hierarchy (we won't use this in this program but this may be helpful for reference)
    std::vector<std::vector<cv::Point> > ptContours1;        // declare a vector for the contours
    std::vector<cv::Vec4i> v4iHierarchy1;         
    std::vector<std::vector<cv::Point> > ptContours2;        // declare a vector for the contours
    std::vector<cv::Vec4i> v4iHierarchy2;  

    cv::findContours(matThreshCopy,             // input image, make sure to use a copy since the function will modify this image in the course of finding contours
        ptContours,                             // output contours
        v4iHierarchy,                           // output hierarchy
        cv::RETR_EXTERNAL,                      // retrieve the outermost contours only
        cv::CHAIN_APPROX_SIMPLE);               // compress horizontal, vertical, and diagonal segments and leave only their end points
        std::cout << "Mat size1 = " << ptContours.size() << std::endl;
    cv::findContours(imgThreshCopy,             // input image, make sure to use a copy since the function will modify this image in the course of finding contours
        ptContours1,                             // output contours
        v4iHierarchy1,                           // output hierarchy
        cv::RETR_EXTERNAL,                      // retrieve the outermost contours only
        cv::CHAIN_APPROX_SIMPLE);               // compress horizontal, vertical, and diagonal segments and leave only their end points
        std::cout << "Mat size2 = " << ptContours1.size() << std::endl;
    cv::findContours(imgThreshCopy1,             // input image, make sure to use a copy since the function will modify this image in the course of finding contours
        ptContours2,                             // output contours
        v4iHierarchy2,                           // output hierarchy
        cv::RETR_EXTERNAL,                      // retrieve the outermost contours only
        cv::CHAIN_APPROX_SIMPLE);               // compress horizontal, vertical, and diagonal segments and leave only their end points
        std::cout << "Mat size8 = " << ptContours2.size() << std::endl;



    for (int i = 0; i < ptContours.size(); i++) {               // for each contour
        ContourWithData contourWithData;                                                    // instantiate a contour with data object
        contourWithData.ptContour = ptContours[i];                                          // assign contour to contour with data
        contourWithData.boundingRect = cv::boundingRect(contourWithData.ptContour);         // get the bounding rect
        contourWithData.fltArea = cv::contourArea(contourWithData.ptContour);               // calculate the contour area
        allContoursWithData.push_back(contourWithData);      
             // add contour with data object to list of all contours with data
    }

     
    for (int i = 0; i < ptContours1.size(); i++) {       
    ContourWithData contourWithData1;
    contourWithData1.ptContour1 = ptContours1[i];  
    contourWithData1.boundingRect1 = cv::boundingRect(contourWithData1.ptContour1);  
    contourWithData1.fltArea1 = cv::contourArea(contourWithData1.ptContour1);
    allContoursWithData1.push_back(contourWithData1);  
                                   // add contour with data object to list cv::contourArea(contourWithData1.ptContour);   
    }
      for (int i = 0; i < ptContours2.size(); i++) {       
    ContourWithData contourWithData2;
    contourWithData2.ptContour2 = ptContours2[i];  
    contourWithData2.boundingRect2 = cv::boundingRect(contourWithData2.ptContour2);  
    contourWithData2.fltArea2 = cv::contourArea(contourWithData2.ptContour2);
    allContoursWithData2.push_back(contourWithData2);  
    }
  

                                    // show ROI image for reference  块访问与操作（ROI区域的选取）
    for (int i = 0; i < allContoursWithData.size(); i++) {                      // for all contours
        if (allContoursWithData[i].checkIfContourIsValid()) {                   // check if valid
            validContoursWithData.push_back(allContoursWithData[i]);            // if so, append to valid contour list
        }
    }
    for (int i = 0; i < allContoursWithData1.size(); i++) {                      // for all contours
        if (allContoursWithData1[i].checkIfContourIsValid1()) {                   // check if valid
            validContoursWithData1.push_back(allContoursWithData1[i]);            // if so, append to valid contour list
        }
    }
  for (int i = 0; i < allContoursWithData2.size(); i++) {                      // for all contours
        if (allContoursWithData2[i].checkIfContourIsValid2()) {                   // check if valid
            validContoursWithData2.push_back(allContoursWithData2[i]);            // if so, append to valid contour list
        }
    }
          std::cout << "Mat size3 = " <<allContoursWithData2.size() << std::endl;
          std::cout << "Mat size valid 5000 = " <<validContoursWithData2.size() << std::endl;
        std::cout << "Mat size5 = " <<allContoursWithData.size() << std::endl;
          std::cout << "Mat size valid test = " <<validContoursWithData.size() << std::endl;
            std::cout << "Mat size5 = " <<allContoursWithData1.size() << std::endl;
          std::cout << "Mat size valid 224 = " <<validContoursWithData1.size() << std::endl;
    // sort contours from left to right
    std::sort(validContoursWithData.begin(), validContoursWithData.end(), ContourWithData::sortByBoundingRectXPosition);
     std::sort(validContoursWithData1.begin(), validContoursWithData1.end(), ContourWithData::sortByBoundingRectXPosition1);
        std::sort(validContoursWithData2.begin(), validContoursWithData2.end(), ContourWithData::sortByBoundingRectXPosition2);

   
    std::string strFinalString;         // declare final string, this will have the final number sequence by the end of the program

    
   

    for (int i = 0; i < validContoursWithData.size(); i++) {            // for each contour

                                                                        // draw a green rect around the current char
        cv::rectangle(matTestingNumbers,                            // draw rectangle on original image
           validContoursWithData[i].boundingRect,        // rect to draw
            cv::Scalar(0, 255, 0),                        // green
            2);                                           // thickness

        cv::Mat matROI = matThresh(validContoursWithData[i].boundingRect);          // get ROI image of bounding rect
        
        cv::Size s = matROI.size();
          
     

        cv::Mat matROIResized;
        cv::resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));     // resize image, this will be more consistent for recognition and storage

        cv::Mat matROIFloat;
        
         cv::Mat matROIResized1 (24,28,CV_8UC1,Scalar(0)) ;
         matROIResized.copyTo(matROIResized1(Rect(4, 4, 20, 20)));
         
        cv::Mat matROIResized2;
        cv::resize(matROIResized1, matROIResized2, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT)); 
         
        matROIResized.convertTo(matROIFloat, CV_32FC1);             // convert Mat to float, necessary for call to find_nearest

      
        testCells.push_back(matROIResized2);

      
       std::cout << "\n\n" << " testCells = " << matROIResized2<< "\n\n"; 

}
  



 for (int i = 0; i < validContoursWithData1.size(); i++) {            // for each contour                                                                 // draw a green rect around the current char
           
           cv::rectangle(imgTrainingletters,                            // draw rectangle on original image
           validContoursWithData1[i].boundingRect1,        // rect to draw
           cv::Scalar(255,255,0),                        // red
            2);      
                                                 // thickness
            cv::Mat matROI1 = imgThresh(validContoursWithData1[i].boundingRect1); 
            cv::Mat matROIResized11;
             cv::resize(matROI1, matROIResized11, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));
       

           trainCells1.push_back(matROIResized11);  




  }

for (int i = 0; i < validContoursWithData2.size(); i++) {            // for each contour                                                                 // draw a green rect around the current char
           
           cv::rectangle(imgTrainingnumbers1,                            // draw rectangle on original image
           validContoursWithData2[i].boundingRect2,        // rect to draw
           cv::Scalar(255,255,0),                        // red
            2);      
                                                 // thickness
            cv::Mat matROI8 = imgThresh1(validContoursWithData2[i].boundingRect2); 
             cv::Mat matROIResized88;
             cv::resize(matROI8, matROIResized88, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));
            cv::imshow("matROIResized88", matROIResized88);  
    
  }

    for(int z=0;z<validContoursWithData2.size();z++){
        if(z % 50 == 0 && z != 0){
            }
      }



    cout << "Image Count : " << ImgCount << endl;
    std::cout << "\n\n"<< "validContoursWithData1 " << validContoursWithData1.size() << "\n\n";


    float digitClassNumber = 0;
    float digitClassNumber1 = 10;
//0.9*
    

 for (int i = 0; i < validContoursWithData.size(); i++) { 
      digitClassNumber = digitClassNumber + 1;
      testLabels.push_back(digitClassNumber);
     }
     digitClassNumber = 0;


      digitClassNumber1 = 10;

for (int i = 0; i < validContoursWithData1.size(); i++) { 

    if(i % 56 == 0 && i != 0){
            digitClassNumber1 = digitClassNumber1 + 1;
            }
     trainLabels1.push_back(digitClassNumber1);

         }
          digitClassNumber1 = 10;


 
    

   for(int z=0;z< validContoursWithData2.size();z++){
        if(z % 500 == 0 && z != 0){
            digitClassNumber = digitClassNumber + 1;
            }
       trainLabels.push_back(digitClassNumber);
    }
    digitClassNumber = 0;
    cv::imshow("results ", matTestingNumbers); 
}



void CreateDeskewedTrainTest(vector<Mat> &deskewedTrainCells,vector<Mat> &deskewedTrainCells1,vector<Mat> &deskewedTestCells,vector<Mat> &deskewedTestCells1,  vector<Mat> &trainCells, vector<Mat> &trainCells1,vector<Mat> &testCells)
    
{
    for(int i=0;i<trainCells.size();i++){

        Mat deskewedImg = deskew(trainCells[i]);
        deskewedTrainCells.push_back(deskewedImg);

    }
    

     for(int i=0;i<trainCells1.size();i++){

        Mat deskewedImg1 = deskew(trainCells1[i]);
        deskewedTrainCells1.push_back(deskewedImg1);

   }

      

    for(int i=0;i<testCells.size();i++){
        if (i<4)
       { Mat deskewedImg = deskew(testCells[i]);
        deskewedTestCells1.push_back(deskewedImg);}  
       else 
       { Mat deskewedImg1 = deskew(testCells[i]);
        deskewedTestCells.push_back(deskewedImg1);}  

       cout << "testCells: " << i << endl;
    }         
}

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
                
void CreateTrainTestHOG(vector<vector<float> > &trainHOG, vector<vector<float> > &testHOG, vector<Mat> &deskewedTrainCells, vector<Mat> &deskewedTestCells){

    for(int y=0;y<deskewedTrainCells.size();y++){
        vector<float> descriptors;
        hog.compute(deskewedTrainCells[y],descriptors);
        trainHOG.push_back(descriptors);
    }


   
    for(int y=0;y<deskewedTestCells.size();y++){
        
        vector<float> descriptors;
        hog.compute(deskewedTestCells[y],descriptors);
        testHOG.push_back(descriptors);
    } 
}

void CreateTrainTestHOG1(vector<vector<float> > &trainHOG1, vector<vector<float> > &testHOG1, vector<Mat> &deskewedTrainCells1, vector<Mat> &deskewedTestCells1){

    for(int y=0;y<deskewedTrainCells1.size();y++){
        vector<float> descriptors;
        hog.compute(deskewedTrainCells1[y],descriptors);
        trainHOG1.push_back(descriptors);
    }

 
   
    for(int y=0;y<deskewedTestCells1.size();y++){
        
        vector<float> descriptors;
        hog.compute(deskewedTestCells1[y],descriptors);
        testHOG1.push_back(descriptors);
    } 
}





void ConvertVectortoMatrix(vector<vector<float> > &trainHOG,  vector<vector<float> > &testHOG, Mat &trainMat, Mat &testMat)
{

    int descriptor_size = trainHOG[0].size();
  //  int descriptor_size1 = trainHOG1[0].size();
 std::cout << "\n\n" << "trainHOG.size()" <<trainHOG.size()<< "\n\n";  
 //std::cout << "\n\n" << "trainHOG1.size() " <<trainHOG1.size()<< "\n\n";  

    for(int i = 0;i<trainHOG.size();i++){
        for(int j = 0;j<descriptor_size;j++){
           trainMat.at<float>(i,j) = trainHOG[i][j]; 
        }
    }

    for(int i = 0;i<testHOG.size();i++){
        for(int j = 0;j<descriptor_size;j++){
            testMat.at<float>(i,j) = testHOG[i][j]; 
        }
    }
}

void ConvertVectortoMatrix1(vector<vector<float> > &trainHOG1,  vector<vector<float> > &testHOG1, Mat &trainMat1, Mat &testMat1)
{

    int descriptor_size1 = trainHOG1[0].size();
    std::cout << "\n\n"<< "descriptor_size1 = " << descriptor_size1<< "\n\n";  
  //  int descriptor_size1 = trainHOG1[0].size();
 std::cout << "\n\n" << "trainHOG1.size()" <<trainHOG1.size()<< "\n\n";  
 //std::cout << "\n\n" << "trainHOG1.size() " <<trainHOG1.size()<< "\n\n";  

    for(int i = 0;i<trainHOG1.size();i++){
        for(int j = 0;j<descriptor_size1;j++){
           trainMat1.at<float>(i,j) = trainHOG1[i][j]; 
        }
    }


    for(int i = 0;i<testHOG1.size();i++){
        for(int j = 0;j<descriptor_size1;j++){
            testMat1.at<float>(i,j) = testHOG1[i][j]; 
        }
    }
}







void getSVMParams(SVM *svm)
{
    cout << "Kernel type     : " << svm->getKernelType() << endl;
    cout << "Type            : " << svm->getType() << endl;
    cout << "C               : " << svm->getC() << endl;
    cout << "Degree          : " << svm->getDegree() << endl;
    cout << "Nu              : " << svm->getNu() << endl;
    cout << "Gamma           : " << svm->getGamma() << endl;
}

Ptr<SVM> svmInit(float C, float gamma)
{
  Ptr<SVM> svm = SVM::create();
  svm->setGamma(gamma);
  svm->setC(C);
  svm->setKernel(SVM::RBF);
  svm->setType(SVM::C_SVC);

  return svm;
}

void svmTrain(Ptr<SVM> svm, Mat &trainMat, vector<int> &trainLabels)
{
  Ptr<TrainData> td = TrainData::create(trainMat, ROW_SAMPLE, trainLabels);

  
  svm->train(td);

  svm->save("3.xml");
}
void svmTrain1(Ptr<SVM> svm, Mat &trainMat1, vector<int> &trainLabels1)
{
  Ptr<TrainData> td1 = TrainData::create(trainMat1, ROW_SAMPLE, trainLabels1);

  
  svm->train(td1);

  svm->save("4.xml");
}





void svmPredict1(Ptr<SVM> svm, Mat &testResponse1, Mat &testMat1 )
{
 static const char ini[4] =  { 'L', 'R', 'F', 'B', };

  svm->predict(testMat1, testResponse1);

//   cv::Size size = testResponse.size();
//   int total = size.width * size.height * testResponse.channels();
//   std::cout << "Mat size = " << total << std::endl;
//  std::vector<uchar> data(testResponse.ptr(), testResponse.ptr() + total);
for(int i = 0; i < testResponse1.rows; i++)
  {
    // cout << testResponse.at<float>(i,0) << " " << testLabels[i] << endl;
    if(testResponse1.at<float>(i,0) == 10)   
  {   std::cout << "\n\n" << "letters read = " <<'L'<< "\n\n";  }
    else if(testResponse1.at<float>(i,0) == 11)   
   {   std::cout << "\n\n" << "letters read = " <<'R'<< "\n\n";  }
    else if(testResponse1.at<float>(i,0) == 12)   
   {   std::cout << "\n\n" << "letters read = " <<'F'<< "\n\n";  }
    else if(testResponse1.at<float>(i,0) == 13)   
   {   std::cout << "\n\n" << "letters read = " <<'B'<< "\n\n";  }

}
}

void svmPredict(Ptr<SVM> svm, Mat &testResponse, Mat &testMat )
{
  svm->predict(testMat, testResponse);


for(int i = 0; i < testResponse.rows; i++)
{ float   matAsString =  testResponse.at<float>(i);

cout << "numbers read1 = "<< endl << " "  <<  matAsString << endl << endl;

}
}




int main()
{
 
 


    vector<Mat> trainCells;
     vector<Mat> trainCells1;
    vector<Mat> testCells;
    vector<Mat> trialcells;
   
    
    
    vector<int> trainLabels;
    vector<int> trainLabels1;

    vector<int> testLabels;
    vector<int> testLabels1;
 

    loadTrainTestLabel(pathName,trainCells,trainCells1,testCells,trialcells,trainLabels,trainLabels1,testLabels);
    vector<Mat> deskewedTrainCells;
    vector<Mat> deskewedTrainCells1;

    vector<Mat> deskewedTestCells;
    vector<Mat> deskewedTestCells1;





    CreateDeskewedTrainTest(deskewedTrainCells,deskewedTrainCells1,deskewedTestCells,deskewedTestCells1,trainCells,trainCells1,testCells);
    
    std::vector<std::vector<float> > trainHOG;
    std::vector<std::vector<float> > trainHOG1;

    std::vector<std::vector<float> > testHOG;
    std::vector<std::vector<float> > testHOG1;

    CreateTrainTestHOG(trainHOG,testHOG,deskewedTrainCells,deskewedTestCells);
    CreateTrainTestHOG1(trainHOG1,testHOG1,deskewedTrainCells1,deskewedTestCells1);

    int descriptor_size = trainHOG[0].size();
    int descriptor_size1 = trainHOG1[0].size();

     
 float   matAsString1 = 11;
  string oss= std::to_string(matAsString1);

    cout << "Descriptor Size : " << descriptor_size << endl;
    cout << "Descriptor Size1 : " << descriptor_size1 << endl;
  

  cout << "sb: "<< oss <<"\n";



    Mat trainMat(trainHOG.size(),descriptor_size,CV_32FC1);
    Mat trainMat1(trainHOG1.size(),descriptor_size1,CV_32FC1);


    Mat testMat(testHOG.size(),descriptor_size,CV_32FC1);
    Mat testMat1(testHOG1.size(),descriptor_size1,CV_32FC1);



    ConvertVectortoMatrix(trainHOG,testHOG,trainMat,testMat);
    ConvertVectortoMatrix1(trainHOG1,testHOG1,trainMat1,testMat1);

    float C = 12.5, gamma = 0.5;

    Mat testResponse;
    Mat testResponse1;

    Ptr<SVM> model = svmInit(C, gamma);
    Ptr<SVM> model1 = svmInit(C, gamma);


    ///////////  SVM Training  ////////////////
    svmTrain(model, trainMat,trainLabels);
    svmTrain(model1, trainMat1,trainLabels1);



    ///////////  SVM Testing  ////////////////
    svmPredict1(model1, testResponse1, testMat1); 
    svmPredict(model, testResponse, testMat); 

    ////////////// Find Accuracy   ///////////
    float count = 0;
    float accuracy = 0 ;
    getSVMParams(model);
    getSVMParams(model1);


      int intChar = cv::waitKey(0);  
     return (0);

 
    
}
