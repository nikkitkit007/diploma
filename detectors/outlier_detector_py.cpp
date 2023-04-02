#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

namespace py = pybind11;
using namespace cv;
using namespace std;

cv::Mat numpy_to_mat(py::array_t<uint8_t>& input) {
    // Get the buffer information from the NumPy array
    py::buffer_info buf = input.request();

    // Get the dimensions of the NumPy array
    int rows = buf.shape[0];
    int cols = buf.shape[1];

    // Create a new cv::Mat object with the same dimensions as the NumPy array
    cv::Mat output(rows, cols, CV_8UC1);

    // Copy the data from the NumPy array to the cv::Mat object
    uint8_t* src_data = static_cast<uint8_t*>(buf.ptr);
    uint8_t* dst_data = output.data;
    for (int i = 0; i < rows; i++) {
        std::memcpy(dst_data, src_data, cols);
        src_data += buf.strides[0];
        dst_data += output.step[0];
    }

    return output;
}


py::list z_score_from_image(py::array_t<uint8_t>& imag, double threshold) {
    // Load the image
    Mat img = numpy_to_mat(imag);

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

py::list outlier_detection_from_image(py::array_t<uint8_t>& imag, double threshold) {
    // Load the image
    Mat img = numpy_to_mat(imag);

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

py::list mehalanobis_distance_from_image(py::array_t<uint8_t>& imag, double threshold) {
    // Load the image
    Mat img = numpy_to_mat(imag);

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


py::list histogram (py::array_t<uint8_t>& imag, double threshold) {
    // Load the image
    Mat img = numpy_to_mat(imag);

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


py::list histogram_from_image (string filename, double threshold) {
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


py::array_t<double> calculate_map_z_score(py::array_t<double>& input) {
    // Get the number of rows and columns in the input array
    auto rows = input.shape(0);
    auto cols = input.shape(1);

    // Allocate a new numpy array to hold the output
    py::array_t<double> output = py::array_t<double>({rows, cols});

    // Get pointers to the data in the input and output arrays
    auto input_ptr = input.data();
    auto output_ptr = output.mutable_data();

    // Compute the mean and standard deviation of the input data
    double sum = 0.0;
    double sum_squares = 0.0;
    for (int i = 0; i < rows * cols; i++) {
        double value = input_ptr[i];
        sum += value;
        sum_squares += value * value;
    }
    double mean = sum / (rows * cols);
    double variance = sum_squares / (rows * cols) - mean * mean;
    double std_dev = std::sqrt(variance);

    // Fill the output array with z-score estimates
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double value = input_ptr[i * cols + j];
            double z_score = std::abs(value - mean) / std_dev;
            output_ptr[i * cols + j] = z_score;
        }
    }

    return output;
}

py::array_t<double> calculate_map_outlier_detection(py::array_t<unsigned char>& img) {
    // Load the image
    Mat img_mat = numpy_to_mat(img);

    // Calculate the median of the image
    Mat img_blur;
    medianBlur(img_mat, img_blur, 3);
    double median = cv::mean(img_blur)[0];

    // Create a copy of the image for highlighting potential attacks
    Mat img_copy = img_mat.clone();

    // Create an output numpy array
    py::array_t<double> result({img_mat.rows, img_mat.cols});

    // Fill the numpy array with difference/median values
    auto buf = result.mutable_unchecked<2>();
    for (int i = 0; i < img_mat.rows; i++) {
        for (int j = 0; j < img_mat.cols; j++) {
            // Define the region of interest
            Rect roi(j - 1, i - 1, 3, 3);

            // Check the boundaries of the ROI and adjust if needed
            roi.x = max(0, roi.x);
            roi.y = max(0, roi.y);
            roi.width = min(img_mat.cols - roi.x, roi.width);
            roi.height = min(img_mat.rows - roi.y, roi.height);

            // Calculate the median of the ROI
            Mat roi_img = img_mat(roi);
            double roi_median = cv::mean(roi_img)[0];

            // Calculate the difference from the median
            double diff = abs(img_mat.at<uchar>(i, j) - roi_median);

            // Calculate the difference/median value
            double val = diff / median;

            // Set the value in the numpy array
            buf(i, j) = val;
        }
    }
    return result;
}

py::array_t<double> calculate_map_histogram(py::array_t<uint8_t>& imag) {
    // Load the image
    Mat img = numpy_to_mat(imag);

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

    // Create a numpy array to hold the deviation values
    py::array_t<double> deviation_array({ img.rows, img.cols });

    // Loop through each pixel in the image and calculate its deviation value
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            int pixelVal = img.at<uchar>(i, j);
            double deviation = abs(pixelVal - mean[0]) / stddev[0];
            deviation_array.mutable_at(i, j) = deviation;
        }
    }

    return deviation_array;
}

py::array_t<double> calculate_map_mehalanobis_distance(py::array_t<uint8_t>& imag) {
    // Load the image
    Mat img = numpy_to_mat(imag);

    // Convert the image to a double-precision floating point matrix
    Mat img_double;
    img.convertTo(img_double, CV_64F);

    // Calculate the mean and covariance matrix of the image
    Mat meanMat, covar;
    calcCovarMatrix(img_double, covar, meanMat, COVAR_NORMAL | COVAR_ROWS);

    // Invert the covariance matrix
    Mat invCovar;
    invert(covar, invCovar, DECOMP_SVD);

    // Create a 2D numpy array to store the deviation values
    int rows = img.rows;
    int cols = img.cols;
    py::array_t<double> deviations = py::array_t<double>({rows, cols});
    auto r = deviations.mutable_unchecked<2>();

    // Loop through each pixel in the image and calculate its Mahalanobis distance
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Mat pixelMat(1, 1, CV_64FC1);
            pixelMat.at<double>(0, 0) = img_double.at<double>(i, j);
            Mat diff = pixelMat - meanMat;
            Mat mdMat = diff * invCovar * diff.t();
            double md = sqrt(mdMat.at<double>(0, 0));
            r(i, j) = md;
        }
    }

    return deviations;
}



PYBIND11_MODULE(outlier_detector, m) {
    m.doc() = "Python interface for detect one pixel attack C++ code";
    m.def("z_score_from_image", &z_score_from_image, "Calculate the z-score of an image",
        py::arg("file"),
        py::arg("threshold") = 3.0);
    m.def("z_score", &z_score, "Calculate the z-score of an image",
        py::arg("filename"),
        py::arg("threshold") = 3.0);
    m.def("outlier_detection", &outlier_detection, "Calculate the outlier detection of an image",
        py::arg("filename"),
        py::arg("threshold") = 3.0);
    m.def("outlier_detection_from_image", &outlier_detection_from_image, "Calculate the outlier detection of an image",
        py::arg("file"),
        py::arg("threshold") = 3.0);
    m.def("mehalanobis_distance", &mehalanobis_distance, "Calculate the mehalanobis distance of an image",
        py::arg("filename"),
        py::arg("threshold") = 3.0);
    m.def("mehalanobis_distance_from_image", &mehalanobis_distance_from_image, "Calculate the mehalanobis distance of an image",
        py::arg("file"),
        py::arg("threshold") = 3.0);
    m.def("histogram", &histogram, "Calculate the histogram of an image",
        py::arg("filename"),
        py::arg("threshold") = 3.0);
    m.def("histogram_from_image", &histogram_from_image, "Calculate the histogram of an image",
        py::arg("file"),
        py::arg("threshold") = 3.0);
    m.def("calculate_map_z_score", &calculate_map_z_score, "Calculate map",
        py::arg("matrix"));
    m.def("calculate_map_outlier_detection", &calculate_map_outlier_detection, "Calculate map",
        py::arg("matrix"));
    m.def("calculate_map_histogram", &calculate_map_histogram, "Calculate map",
        py::arg("matrix"));
    m.def("calculate_map_mehalanobis_distance", &calculate_map_mehalanobis_distance, "Calculate map",
        py::arg("matrix"));

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
