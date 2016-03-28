#include "opencv2/objdetect.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/utility.hpp"

#include "opencv2/videoio/videoio_c.h"
#include "opencv2/highgui/highgui_c.h"

#include <cctype>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include "dirent.h"

using namespace std;
using namespace cv;


void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale, string& filename );

string cascadeName = "haarcascade_frontalface_alt.xml"; //10+ms 
//string cascadeName = "./lbpcascade_frontalface.xml"; //2.5ms 
int main( int argc, const char** argv )
{
    CvCapture* capture = 0;
    Mat frame, frameCopy, image;
    string inputName;
    bool tryflip = false;
    CascadeClassifier cascade, nestedCascade;
    double scale = 1;

    if( !cascade.load( cascadeName ) )
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        return -1;
    }

    if (!argv[1])
    {
        printf ("Please, specify input folder, containing images, as an argument!\n");
        return 1;
    }
    std::string inputDirectory = argv[1];
    struct dirent **namelist;
    int n = scandir(inputDirectory.c_str(), &namelist, 0, alphasort);
    if (n < 0)
        perror("scandir");

    for (int count = 0; count < n; count++)
    {
        std::string filepath =
          inputDirectory + "/" + std::string (namelist[count]->d_name);
        std::string filename = std::string (namelist[count]->d_name);

        image = cv::imread (filepath.c_str ());

        if(image.empty()) cout << "Couldn't read image" << endl;

        cvNamedWindow( "result", 1 );
        cout << "In image read" << endl;
        
        if( !image.empty() )
        {
                
            detectAndDraw( image, cascade, nestedCascade, scale, filename );
            
        }
        int c = waitKey(0);
        if (c == 27)
        {
            break;
        }
    }

}

void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale, string& filename)
{
    int i = 0;
    double t = 0;
    vector<Rect> faces, faces2;
    Mat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );
    const static Scalar colors[] =  { CV_RGB(0,0,255),
        CV_RGB(0,128,255),
        CV_RGB(0,255,255),
        CV_RGB(0,255,0),
        CV_RGB(255,128,0),
        CV_RGB(255,255,0),
        CV_RGB(255,0,0),
        CV_RGB(255,0,255)} ;
    cvtColor( img, gray, COLOR_BGR2GRAY );
    resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );

    t = (double)cvGetTickCount();

    cascade.detectMultiScale( smallImg, faces,
        1.1, 3, 0
        //|CASCADE_FIND_BIGGEST_OBJECT
        //|CASCADE_DO_ROUGH_SEARCH
        |CASCADE_SCALE_IMAGE
        ,
        Size(100, 100) );
    t = (double)cvGetTickCount() - t;
    printf( "detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
    for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ )
    {
    	Scalar color = colors[i%8];
        
        Mat face(img, Rect(cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
                       cvPoint(cvRound((r->x + r->width-1)*scale), cvRound((r->y + r->height-1)*scale))));

        /*Mat earL(img, Rect(cvPoint(cvRound(r->x*scale - r->width*1/2), cvRound(r->y*scale + r->height*1/3)),
                       cvPoint(cvRound(r->x*scale), cvRound((r->y + r->height*4/3)*scale))));*/

        /*Mat earR(img, Rect(cvPoint(cvRound(r->x*scale + r->width -1 ), cvRound(r->y*scale + r->height*1/3)),
                       cvPoint(cvRound((r->x + r->width*3/2)*scale), cvRound((r->y + r->height*4/3)*scale))));

        Mat chest(img, Rect(cvPoint(cvRound(r->x*scale - r->width*1/2), cvRound((r->y + r->height)*scale)),
                       cvPoint(cvRound((r->x + r->width*3/2)*scale), cvRound((r->y + r->height*2)*scale))));*/

        rectangle( img, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
                       cvPoint(cvRound((r->x + r->width-1)*scale), cvRound((r->y + r->height-1)*scale)),
                       color, 1, 8, 0);  //face

        rectangle( img, cvPoint(cvRound((r->x + r->width*13/100.0)*scale ), cvRound((r->y + r->height*25/100.0)*scale)),
                       cvPoint(cvRound((r->x + r->width*48/100.0)*scale), cvRound((r->y + r->height*55/100.0)*scale)),
                       color, 1, 8, 0);  //Leye

        rectangle( img, cvPoint(cvRound((r->x + r->width*52/100.0)*scale ), cvRound((r->y + r->height*25/100.0)*scale)),
                       cvPoint(cvRound((r->x + r->width*87/100.0)*scale), cvRound((r->y + r->height*55/100.0)*scale)),
                       color, 1, 8, 0);  //Reye


        cv::imshow( "result", img);
   
        while(1){
            cout << "f(face) r(earR) l(earL) c(chest) for saving ROI and n for next image!" << endl;
            char c = waitKey(0);
            switch(c){
                
                case 'f': cout << "save found faces" << endl;
                imwrite("./face/face_" + std::string(filename), face);

                break;
                /*case 'r':
                imwrite("./earR/earR_" + std::string(filename), earR);
                case 'l':
                imwrite("./earL/earL_" + std::string(filename), earL);
                case 'c':
                imwrite("./chest/chest_" + std::string(filename), chest);
                break;*/

            }
            if(c=='n') break;
            
        }
        
       cout << "next image!" << endl;
    }
}
