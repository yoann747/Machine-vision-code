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

int main()
{
Mat e;
Mat f;
Mat g;
Mat a=imread("L.jpg");  
Mat b=imread("R.jpg");  
Mat c=imread("F.jpg");  
Mat d=imread("B.jpg");  
    vconcat(a,b,e);  
     vconcat(c,d,f); 
     vconcat(e,f,g); 
imwrite("letter1.png",g);


    }