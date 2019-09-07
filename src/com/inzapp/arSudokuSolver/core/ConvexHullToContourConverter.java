package com.inzapp.arSudokuSolver.core;

import com.inzapp.arSudokuSolver.config.Config;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class ConvexHullToContourConverter {
    /**
     * @param progress
     * @param sudokuContour
     * @return
     */
    public MatOfPoint convert(Mat progress, MatOfPoint sudokuContour) {
        if (!Config.VIEW_PROGRESS)
            return null;

        try {
            // convex hull
            MatOfInt hull = new MatOfInt();
            Imgproc.convexHull(sudokuContour, hull);

            // get hull idx list
            Point[] contourArray = sudokuContour.toArray();
            Point[] hullPoints = new Point[hull.rows()];

            // copy hull point to array
            List<Integer> hullContourIdxList = hull.toList();
            for (int i = 0; i < hullContourIdxList.size(); i++)
                hullPoints[i] = contourArray[hullContourIdxList.get(i)];

            // convert hull to list
            List<MatOfPoint> hullList = new ArrayList<>();
            hullList.add(new MatOfPoint(hullPoints));

            // draw hull
            Imgproc.drawContours(progress, hullList, 0, new Scalar(0, 255, 0), 2);

            return new MatOfPoint(hullPoints);
        } catch (Exception e) {
            return null;
        }
    }
}
