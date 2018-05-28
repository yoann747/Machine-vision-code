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
#include <iomanip>


using namespace cv::ml;
using namespace cv;
using namespace std;




std::string strFinalString;  

string pathName = "digits.png";
string pathName1 = "digitnumbers.png";
cv::Mat matTestingNumbers = cv::imread("digitnumbers.png");  

int main()
{


  Mat combine,combine1,combine2;  
  Mat img;
  Mat temp;
  temp = imread("1.png"); 
  Mat temp1;
  temp1 = imread("28.png"); 
  Mat temp2;
  Mat c;
  Mat d;
  int i;
//    Mat a=imread("1.png");  
//    Mat b=imread("2.png");  
//     // Mat c=imread("3.png");  
//     // Mat d=imread("4.png");  
//     hconcat(a,b,c);  
//     hconcat(a,c,d); 
// imshow("d",d);  
//     //hconcat(c,d,combine2);  
//     // vconcat(combine1,combine2,combine);  
//     //namedWindow("Combine",CV_WINDOW_AUTOSIZE);  
//     //imshow("Combine",combine);  
//     //imwrite("test.jpg",combine); // A JPG FILE IS BEING SAVED  
//     //waitKey(0);  
    std::string folder = "/home/xiaotian/English/Hnd/Img/Sample011";
    std::string suffix = ".png";
    int counter = 0;

    cv::Mat myImage;

    // while (1)
    // {
    //     std::stringstream ss;
    //     ss << std::setw(4) << std::setfill('0') << counter; // 0000, 0001, 0002, etc...
    //     std::string number = ss.str();

    //     std::string name = folder + number + suffix;
    //     myImage = cv::imread(name);

    //     cv::imshow("HEYO", myImage);
    //     int c = cv::waitKey(1);

    //     counter++;
    // }
    // return 0;




    for (i=1;i<28;i++)
    {
    std::stringstream ss;
     ss << std::setw(1) << i; // 0000, 0001, 0002, etc...
     std::string number = ss.str();
      std::string name = number + suffix;
     Mat img = imread(name); 
     
     
     hconcat(temp,img,temp);  
     
    //Mat img = imread("i.png");
   
    //imwrite("test.jpg",combine)
    cout<<"i=:"<<i<<endl;  
    //cout<<"img=:"<<img<<endl;  
    //std::cout << "\n\n" << "img=: " << img << "\n\n"; 
    //cv::imshow("HEYO", temp);
    //cv::imshow("HEYO", img);
    //int c = cv::waitKey(0);
    
  //int f = cv::waitKey(0);
  if (i==27)
  {   imwrite("letter0001.jpg",temp) ;   }
   
   }
  
    for (i=28;i<55;i++)
    {
    std::stringstream ss;
     ss << std::setw(1) << i; // 0000, 0001, 0002, etc...
     std::string number = ss.str();
      std::string name = number + suffix;
     Mat img = imread(name); 
     
     
     hconcat(temp1,img,temp1);  
     
    //Mat img = imread("i.png");
   
    //imwrite("test.jpg",combine)
    cout<<"i=:"<<i<<endl;  
    //cout<<"img=:"<<img<<endl;  
    //std::cout << "\n\n" << "img=: " << img << "\n\n"; 
    //cv::imshow("HEYO", temp);
    //cv::imshow("HEYO", img);
    //int c = cv::waitKey(0);
    
  //int f = cv::waitKey(0);
  if (i==54)
  {   imwrite("letter0002.jpg",temp1) ;   }
   
   }

 vconcat(temp,temp1,temp2);
 imwrite("A.jpg",temp2) ;
     return 0;  
}