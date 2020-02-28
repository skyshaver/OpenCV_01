#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define CVUI_IMPLEMENTATION
#include <cvui.h>

#include <iostream>
#include <chrono>
#include <algorithm>
#include <thread>

using namespace std::chrono_literals;


//using namespace std;
//using namespace cv;
using std::cout; using std::endl;


void Sharpen(const cv::Mat& myImage, cv::Mat& Result);

void ShowFiltersOnImages();

cv::Mat BrightenContrast(cv::Mat& image, double alpha, int beta)
{
    cv::Mat newImage = cv::Mat::zeros(image.size(), image.type());
    image.convertTo(newImage, -1, alpha, beta);
    /*for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            for (int c = 0; c < image.channels(); c++) {
                newImage.at<cv::Vec3b>(y, x)[c] =
                    cv::saturate_cast<uchar>(alpha * image.at<cv::Vec3b>(y, x)[c] + beta);
            }
        }
    }*/
    return newImage;
}

int main(int argc, char* argv[])
{
    //ShowFiltersOnImages();
    const char* videoPath = "videos\\butterfly.mp4";
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened())
    {
        cout << "Error::video opening file";
        std::exit(-1);
    }

    // calculate frame rate
    double fps = cap.get(cv::CAP_PROP_FPS);
    double frameRate = 1000 / fps;
    int frameHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int frameWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    std::cout << fps << ' ' << frameRate << '\n';

     /*cv::namedWindow("Original", cv::WINDOW_NORMAL); 
     cv::resizeWindow("Original", frameWidth / 2, frameHeight / 2);*/

    cv::namedWindow("Effect", cv::WINDOW_NORMAL);
    cv::resizeWindow("Effect", frameWidth / 2, frameHeight / 2);
    cv::RNG_MT19937 rng(0xFFFFFFFF);
    
    static auto startTime = std::chrono::high_resolution_clock::now();
    
    /*const char* uiWindow = "cvui";
    cv::Mat uiFrame = cv::Mat(200, 500, CV_8UC3);
    cvui::init(uiWindow);*/

    while(true)
    {
        
        cv::Mat frame;
        cap >> frame;
        //imshow("Original", frame);
        

        // time
        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
        double alpha = double(sin(time)) + 2;

        // int beta = rng.uniform(80, 120);
        // double alpha = rng.uniform(1.5, 2.5);
        static int beta = 0;
        cv::Mat dst = BrightenContrast(frame, alpha, beta);
        //Sharpen(frame, dst);

        imshow("Effect", dst);

        // press ESC to exit
        char c = (char)cv::waitKey(frameRate);
        if (c == 27) { break; }

        // ui window
        /*uiFrame = cv::Scalar(49, 52, 49);
        cvui::trackbar(uiFrame,40, 30, 220, &beta, int(0), int(100));
        cvui::imshow(uiWindow, uiFrame);
        frameCount++;*/
    }
    
    cap.release();
    
    cv::destroyAllWindows();

    return EXIT_SUCCESS;
}


//! [basic_method]
void Sharpen(const cv::Mat& myImage, cv::Mat& Result)
{
    //! [8_bit]
    CV_Assert(myImage.depth() == CV_8U);  // accept only uchar images
  //! [8_bit]

  //! [create_channels]
    const int nChannels = myImage.channels();
    Result.create(myImage.size(), myImage.type());
    //! [create_channels]

    //! [basic_method_loop]
    for (int j = 1; j < myImage.rows - 1; ++j)
    {
        const uchar* previous = myImage.ptr<uchar>(j - 1);
        const uchar* current = myImage.ptr<uchar>(j);
        const uchar* next = myImage.ptr<uchar>(j + 1);

        uchar* output = Result.ptr<uchar>(j);

        for (int i = nChannels; i < nChannels * (myImage.cols - 1); ++i)
        {
            *output++ = cv::saturate_cast<uchar>(5 * current[i]
                - current[i - nChannels] - current[i + nChannels] - previous[i] - next[i]);
        }
    }
    //! [basic_method_loop]

    //! [borders]
    Result.row(0).setTo(cv::Scalar(0));
    Result.row(Result.rows - 1).setTo(cv::Scalar(0));
    Result.col(0).setTo(cv::Scalar(0));
    Result.col(Result.cols - 1).setTo(cv::Scalar(0));
    //! [borders]
}
//! [basic_method]
void SharpenInternal(cv::Mat& src, cv::Mat& dst)
{
    //![kern] AKAK mask
    cv::Mat sharpenMask = (cv::Mat_<char>(3, 3) << 0, -1, 0,
        -1, 5, -1,
        0, -1, 0);
    //![kern]

    //![filter2D]
    filter2D(src, dst, src.depth(), sharpenMask);
    //![filter2D]
}


void ShowFiltersOnImages()
{
    const char* filename_01 = "images\\old_geo.jpg";
    const char* filename_02 = "images\\old_geo_02.jpg";
    cv::Mat src, src_02, dst0, dst1, dst2;

    src = cv::imread(cv::samples::findFile(filename_01), cv::IMREAD_COLOR);
    src_02 = cv::imread(cv::samples::findFile(filename_02), cv::IMREAD_COLOR);

    if (src.empty())
    {
        std::cerr << "Can't open image [" << filename_01 << "]" << endl;
        std::exit(-1);
    }
    if (src_02.empty())
    {
        std::cerr << "Can't open image [" << filename_02 << "]" << endl;
        std::exit(-1);
    }
    cv::namedWindow("Input 1", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Input 2", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Sharpen 1", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Sharpen 2", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Blend", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("BrightContrast", cv::WINDOW_AUTOSIZE);

    cv::imshow("Input 1", src);
    cv::imshow("Input 2", src_02);

    Sharpen(src, dst0);
    Sharpen(src_02, dst1);
    cv::imshow("Sharpen 1", dst0);
    cv::imshow("Sharpen 2", dst1);

    // blend
    double alpha = 0.5, beta;
    beta = (1.0 - alpha);
    cv::addWeighted(dst0, alpha, dst1, beta, 0.0, dst2);
    cv::imshow("Blend", dst2);

    cv::Mat newImage = BrightenContrast(dst2, 1.0, 75);
    cv::imshow("BrightContrast", newImage);



    cv::waitKey();
}