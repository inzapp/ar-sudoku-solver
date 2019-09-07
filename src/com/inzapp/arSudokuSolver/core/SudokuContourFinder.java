package com.inzapp.arSudokuSolver.core;

import com.inzapp.arSudokuSolver.config.Config;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import java.util.ArrayList;
import java.util.List;

public class SudokuContourFinder {
    /**
     * find contour of sudoku area using findContours() method
     *
     * @param progress mat for viewing progress
     * @param proc     processed image
     * @return contour of sudoku area
     */
    public MatOfPoint find(Mat progress, Mat proc) {
        // find contours
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(proc, contours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_NONE);

        // calculate center of raw img
        Point frameCenter = new Point(progress.width() / 2, progress.height() / 2);

        // extract sudoku contour
        double maxArea = 0;
        MatOfPoint sudokuContour = new MatOfPoint();
        int rawResolution = progress.rows() * progress.cols();
        for (MatOfPoint contour : contours) {
            Rect rect = Imgproc.boundingRect(contour);

            // filter : area
            if (!(maxArea < rect.area()))
                continue;

            // filter : area ratio
            if (rect.area() < rawResolution * 0.2 || rawResolution * 0.8 < rect.area())
                continue;

            Point contourCenter = getCenterPoint(contour);
            double distanceFromCenter = getDistance(frameCenter, contourCenter);

            // filter : distance of contour center and frame center
            if (100 < distanceFromCenter)
                continue;

            // save
            maxArea = rect.area();
            sudokuContour = contour;
        }

        Point contourCenter = getCenterPoint(sudokuContour);

        if (Config.VIEW_PROGRESS) {
            Imgproc.circle(progress, contourCenter, 10, new Scalar(0, 0, 255), -1);
            Imgproc.circle(progress, frameCenter, 10, new Scalar(255, 0, 0), -1);
        }

        return sudokuContour;
    }

    /**
     * calculate the Euclidean distance from the two points
     *
     * @param a point 1
     * @param b another point
     * @return Euclidean distance of point a and b
     */
    private double getDistance(Point a, Point b) {
        return Core.norm(new MatOfPoint(a), new MatOfPoint(b));
    }

    /**
     * calculate center point of contour using moments
     *
     * @param sudokuContour sudoku contour
     * @return center point of contour
     */
    private Point getCenterPoint(MatOfPoint sudokuContour) {
        Moments moments = Imgproc.moments(sudokuContour, false);
        int x = (int) (moments.get_m10() / moments.get_m00());
        int y = (int) (moments.get_m01() / moments.get_m00());
        return new Point(x, y);
    }
}
