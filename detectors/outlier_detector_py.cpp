#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

namespace py = pybind11;
using namespace cv;
using namespace std;

py::list z_score(string filename, double threshold) {
    // Load the image
    Mat img = imread(filename, IMREAD_GRAYSCALE);

    // Calculate the mean and standard deviation of the image
    Scalar mean, stddev;
    meanStdDev(img, mean, stddev);

    // Create a copy of the image for highlighting potential attacks
    Mat img_copy = img.clone();
    py::list candidate_pixels;

    // Loop through each pixel in the image and calculate its Z-score
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            double zscore = (img.at<uchar>(i, j) - mean[0]) / stddev[0];
            if (abs(zscore) > threshold) {
                py::list pixel;
                pixel.append(j); pixel.append(i);
                candidate_pixels.append(pixel);
                // Pixel is an outlier, indicating a potential single pixel attack
                // circle(img_copy, Point(j, i), 2, Scalar(0, 0, 255), -1);
            }
        }
    }

    // Create a NumPy array from the image data
    // py::list result({ img.rows, img.cols }, img.data);

    return candidate_pixels;
}

py::list outlier_detection(string filename, double threshold) {
    // Load the image
    Mat img = imread(filename, IMREAD_GRAYSCALE);

    // Calculate the median of the image
    Mat img_blur;
    medianBlur(img, img_blur, 3);
    double median = cv::mean(img_blur)[0];

    // Create a copy of the image for highlighting potential attacks
    Mat img_copy = img.clone();

    py::list candidate_pixels;

    // Loop through each pixel in the image and calculate its difference from the median
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            // Define the region of interest
            Rect roi(j - 1, i - 1, 3, 3);

            // Check the boundaries of the ROI and adjust if needed
            roi.x = max(0, roi.x);
            roi.y = max(0, roi.y);
            roi.width = min(img.cols - roi.x, roi.width);
            roi.height = min(img.rows - roi.y, roi.height);

            // Calculate the median of the ROI
            Mat roi_img = img(roi);
            double roi_median = cv::mean(roi_img)[0];

            // Calculate the difference from the median
            double diff = abs(img.at<uchar>(i, j) - roi_median);

            // Check if the difference is above the threshold
            if (diff > threshold * median) {
                // Pixel is an outlier, indicating a potential single pixel attack
                py::list pixel;
                pixel.append(j); pixel.append(i);
                candidate_pixels.append(pixel);
            }
        }
    }
    return candidate_pixels;
}

py::list mehalanobis_distance(string filename, double threshold) {
    // Load the image
    Mat img = imread(filename, IMREAD_GRAYSCALE);

    // Convert the image to a double-precision floating point matrix
    Mat img_double;
    img.convertTo(img_double, CV_64F);

    // Calculate the mean and covariance matrix of the image
    Mat meanMat, covar;
    calcCovarMatrix(img_double, covar, meanMat, COVAR_NORMAL | COVAR_ROWS);

    // Invert the covariance matrix
    Mat invCovar;
    invert(covar, invCovar, DECOMP_SVD);

    // Define the threshold for the Mahalanobis distance
    // double threshold = std::stod(argv[2]);

    // Create a copy of the image for highlighting potential attacks
    Mat img_copy = img.clone();
    py::list candidate_pixels;

    // Loop through each pixel in the image and calculate its Mahalanobis distance
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            Mat pixelMat(1, 1, CV_64FC1);
            pixelMat.at<double>(0, 0) = img_double.at<double>(i, j);
            Mat diff = pixelMat - meanMat;
            Mat mdMat = diff * invCovar * diff.t();
            double md = sqrt(mdMat.at<double>(0, 0));
            if (md > threshold) {
                // Pixel is an outlier, indicating a potential single pixel attack
                py::list pixel;
                pixel.append(j); pixel.append(i);
                candidate_pixels.append(pixel);
            }
        }
    }

    return candidate_pixels;

}

py::list histogram (string filename, double threshold) {
    // Load the image
    Mat img = imread(filename, IMREAD_GRAYSCALE);

    // Create a histogram of the image
    int histSize = 256;
    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange = { range };
    bool uniform = true, accumulate = false;
    Mat hist;
    calcHist(&img, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

    // Find the mean and standard deviation of the histogram
    Scalar mean, stddev;
    meanStdDev(hist, mean, stddev);

    // Create a copy of the image for highlighting potential attacks
    Mat img_copy = img.clone();

    py::list candidate_pixels;

    // Loop through each pixel in the image and calculate its value in the histogram
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            int pixelVal = img.at<uchar>(i, j);
            double deviation = abs(pixelVal - mean[0]) / stddev[0];
            if (deviation > threshold) {
                // Pixel is an outlier, indicating a potential single pixel attack
                py::list pixel;
                pixel.append(j); pixel.append(i);
                candidate_pixels.append(pixel);
            }
        }
    }
    return candidate_pixels;
}



PYBIND11_MODULE(outlier_detector, m) {
    m.doc() = "Python interface for detect one pixel attack C++ code";
    m.def("z_score", &z_score, "Calculate the z-score of an image",
        py::arg("filename"),
        py::arg("threshold") = 3.0);
    m.def("outlier_detection", &outlier_detection, "Calculate the outlier detection of an image",
        py::arg("filename"),
        py::arg("threshold") = 3.0);
    m.def("mehalanobis_distance", &mehalanobis_distance, "Calculate the mehalanobis distance of an image",
        py::arg("filename"),
        py::arg("threshold") = 3.0);
    m.def("histogram", &histogram, "Calculate the histogram of an image",
        py::arg("filename"),
        py::arg("threshold") = 3.0);
}
/*
extern "C" PYBIND11_EXPORT PyObject* PyInit_outlier_detector() {
    // Define the module
    py::module m("outlier_detector", "Python module for outlier detection");

    m.def("z_score", &z_score, "Calculate the z-score of an image",
        py::arg("filename"),
        py::arg("threshold") = 3.0);
    m.def("outlier_detection", &outlier_detection, "Calculate the outlier detection of an image",
        py::arg("filename"),
        py::arg("threshold") = 3.0);
    m.def("mehalanobis_distance", &mehalanobis_distance, "Calculate the mehalanobis distance of an image",
        py::arg("filename"),
        py::arg("threshold") = 3.0);
    // Return the module
    return m.ptr();
}*/
