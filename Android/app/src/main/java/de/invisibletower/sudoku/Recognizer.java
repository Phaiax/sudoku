package de.invisibletower.sudoku;

import android.provider.ContactsContract;
import android.util.Log;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;
import org.opencv.photo.Photo;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import static java.lang.Math.max;
import static org.opencv.core.CvType.CV_32S;
import static org.opencv.core.CvType.CV_32SC1;
import static org.opencv.core.CvType.CV_8UC1;

/**
 * Created by daniel on 31.07.17.
 */

public class Recognizer {
    private static final String TAG = "Sudoku::Recognizer";

    private Mat mDebug;

    public Recognizer() {
        mDebug = Mat.eye(10, 10, CV_8UC1);
    }

    private static final int RESIZE_LONG_EDGE = 400;

    public void recognize(Mat gray) {
        Mat resized = resize(gray);
        Mat thresholded = improveAndThreshold(resized);
        List<MatOfPoint> contours = globalSquaredContours(thresholded);

        mDebug = mResized.clone();
        Imgproc.drawContours(mDebug, contours, -1, new Scalar(255), 1);

        // mDebug = thresholded.clone();
        Imgproc.circle(mDebug, new Point(200,200), 100, new Scalar(255), 10);
    }

    private double calcScale(Size shape) {
        return ((double) RESIZE_LONG_EDGE) /  max(shape.width, shape.height);
    }

    private Mat mResized = Mat.zeros(1, 1, CV_8UC1);

    private Mat resize(Mat src) {
        Size src_size = src.size();
        double scale = calcScale(src_size);
        int w = (int) (src_size.width * scale);
        int h = (int) (src_size.height * scale);
        mResized.create(h, w, CV_8UC1); // resizes only if necessary
        Log.d(TAG, String.format("W %d, H %d mR: W %f, H %f", w, h, mResized.size().width, mResized.size().height));
        Imgproc.resize(src, mResized, new Size(w, h));
        return mResized;
    }

    private Mat mEquihist = Mat.zeros(10, 10, CV_8UC1);
    private Mat mDenoise = Mat.zeros(10, 10, CV_8UC1);
    private Mat mThresholded = Mat.zeros(10, 10, CV_8UC1);

    private static final int DENOISE_H =16;
    private static final int DENOISE_TEMPLATE_WINDOW_SIZE = 7;
    private static final int DENOISE_SEARCH_WINDOW_SIZE = 10;
    private static final int THRESHOLD_BLOCK_SIZE = 11;
    private static final int THRESHOLD_ADDED_CONSTANT = 20;

    private Mat improveAndThreshold(Mat src) {
        CLAHE clahe = Imgproc.createCLAHE(2.0, new Size(8, 8));
        mEquihist.create(src.size(), CV_8UC1);
        clahe.apply(src, mEquihist);
        Photo.fastNlMeansDenoising(mEquihist, mDenoise, DENOISE_H,
                                   DENOISE_TEMPLATE_WINDOW_SIZE,
                                   DENOISE_SEARCH_WINDOW_SIZE);
        Imgproc.adaptiveThreshold(mDenoise, mThresholded,
                255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                Imgproc.THRESH_BINARY_INV, THRESHOLD_BLOCK_SIZE,
                THRESHOLD_ADDED_CONSTANT);
        return mThresholded;
    }

    private static final int USE_BIGGEST_N_SQUARES = 6;
    private static final double EPSILON = 5;

    private List<MatOfPoint> globalSquaredContours(Mat thresholded) {
        int mode = Imgproc.RETR_TREE;
        int method = Imgproc.CHAIN_APPROX_SIMPLE;
        List<MatOfPoint> contours = new ArrayList<>(40);
        Mat hierarchy = new Mat();
        Imgproc.findContours(thresholded, contours, hierarchy, mode, method);
        class WithArea implements Comparable<WithArea> {
            public WithArea(MatOfPoint c) {
                this.c = c;
                this.area = Imgproc.contourArea(c);
            }
            public MatOfPoint c;
            public double area;
            public int compareTo(WithArea o) {
                if (o.area - area < 0) {
                    return -1;
                } else  if (o.area == area) {
                    return 0;
                } else  {
                    return 1;
                }
            }
        }
        List<WithArea> sorted = new ArrayList<>(contours.size());
        for (MatOfPoint c: contours) {
            sorted.add(new WithArea(c));
        }
        Collections.sort(sorted);
        List<MatOfPoint> filtered = new ArrayList<>(USE_BIGGEST_N_SQUARES);
        for (int i = 0; i < sorted.size(); i++) {
//            MatOfInt hull = new MatOfInt();
//            Imgproc.convexHull(sorted.get(i).c, hull);
//            MatOfPoint2f hull2 = new MatOfPoint2f();
//            hull.convertTo(hull2, CvType.CV_32FC2);
//            Log.d(TAG, "hull2 " + hull2.depth() + hull2.toString());
//            Mat approxed = Mat.zeros(0, 2, CvType.CV_32F);
//            MatOfPoint2f approxed2 = new MatOfPoint2f(approxed);
//            // hull.convertTo(approxed, CvType.CV_32FC2);
//            Log.d(TAG, "approxed " + approxed.depth() + approxed.toString());
//            Log.d(TAG, "approxed2 " + approxed2.depth() + approxed2.toString());
//            Imgproc.approxPolyDP(hull2, approxed2, EPSILON, true);
//            if (approxed.rows() == 4) {
//                MatOfPoint f = new MatOfPoint();
//                approxed.convertTo(f, CvType.CV_32SC1);
//                filtered.add(new MatOfPoint(f));
//                if (filtered.size() == USE_BIGGEST_N_SQUARES) {
//                    break;
//                }
//            }
        }
        return filtered;
    }

    public Mat getDebugImage() {
        return mDebug;
    }
}
