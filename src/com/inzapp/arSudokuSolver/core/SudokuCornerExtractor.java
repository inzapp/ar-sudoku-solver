package com.inzapp.arSudokuSolver.core;

import com.inzapp.arSudokuSolver.config.Config;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.Arrays;
import java.util.Comparator;

public class SudokuCornerExtractor {
    /**
     * used for calculating 4 corner of sudoku contour
     */
    private static class PointRank {
        int rank = 0;
        Point point;

        PointRank(Point point) {
            this.point = point;
        }
    }

    /**
     * find 4 corner of sudoku contour to perspective transform
     * copy points to point rank and sort
     *
     * @param sudokuContour
     * @param progress
     * @return
     */
    public Point[] extract(MatOfPoint sudokuContour, Mat progress) {
        try {
            Point[] points = sudokuContour.toArray();
            PointRank[] pointRanks = new PointRank[points.length];
            for (int i = 0; i < pointRanks.length; i++)
                pointRanks[i] = new PointRank(points[i]);

            // top left
            resetPointRanks(pointRanks);
            Point topLeft = getTopLeft(pointRanks);

            // top right
            resetPointRanks(pointRanks);
            Point topRight = getTopRight(pointRanks);

            // bottom left
            resetPointRanks(pointRanks);
            Point bottomLeft = getBottomLeft(pointRanks);

            // bottom right
            resetPointRanks(pointRanks);
            Point bottomRight = getBottomRight(pointRanks);

            if (Config.VIEW_PROGRESS) {
                Imgproc.circle(progress, topLeft, 10, new Scalar(0, 0, 255), 2);
                Imgproc.circle(progress, topRight, 10, new Scalar(0, 0, 255), 2);
                Imgproc.circle(progress, bottomLeft, 10, new Scalar(0, 0, 255), 2);
                Imgproc.circle(progress, bottomRight, 10, new Scalar(0, 0, 255), 2);
            }
            return new Point[]{topLeft, topRight, bottomLeft, bottomRight};
        } catch (Exception e) {
            return null;
        }
    }

    /**
     * @param pointRanks
     */
    private void resetPointRanks(PointRank[] pointRanks) {
        for (int i = 0; i < pointRanks.length; i++)
            pointRanks[i].rank = 0;
    }

    /**
     * @param pointRanks
     * @return
     */
    private Point getTopLeft(PointRank[] pointRanks) {
        Arrays.sort(pointRanks, Comparator.comparingDouble(a -> a.point.y));
        for (int i = 0; i < pointRanks.length; i++)
            pointRanks[i].rank += i;
        Arrays.sort(pointRanks, Comparator.comparingDouble(a -> a.point.x));
        for (int i = 0; i < pointRanks.length; i++)
            pointRanks[i].rank += i;
        Arrays.sort(pointRanks, Comparator.comparingInt(a -> a.rank));
        Point topLeft = pointRanks[0].point;
        return topLeft;
    }

    /**
     * @param pointRanks
     * @return
     */
    private Point getTopRight(PointRank[] pointRanks) {
        Arrays.sort(pointRanks, (a, b) -> Double.compare(b.point.x, a.point.x));
        for (int i = 0; i < pointRanks.length; i++)
            pointRanks[i].rank += i;
        Arrays.sort(pointRanks, Comparator.comparingDouble(a -> a.point.y));
        for (int i = 0; i < pointRanks.length; i++)
            pointRanks[i].rank += i;
        Arrays.sort(pointRanks, Comparator.comparingInt(a -> a.rank));
        Point topRight = pointRanks[0].point;
        return topRight;
    }

    /**
     * @param pointRanks
     * @return
     */
    private Point getBottomLeft(PointRank[] pointRanks) {
        Arrays.sort(pointRanks, Comparator.comparingDouble(a -> a.point.x));
        for (int i = 0; i < pointRanks.length; i++)
            pointRanks[i].rank += i;
        Arrays.sort(pointRanks, (a, b) -> Double.compare(b.point.y, a.point.y));
        for (int i = 0; i < pointRanks.length; i++)
            pointRanks[i].rank += i;
        Arrays.sort(pointRanks, Comparator.comparingInt(a -> a.rank));
        Point bottomLeft = pointRanks[0].point;
        return bottomLeft;
    }

    /**
     * @param pointRanks
     * @return
     */
    private Point getBottomRight(PointRank[] pointRanks) {
        Arrays.sort(pointRanks, (a, b) -> Double.compare(b.point.x, a.point.x));
        for (int i = 0; i < pointRanks.length; i++)
            pointRanks[i].rank += i;
        Arrays.sort(pointRanks, (a, b) -> Double.compare(b.point.y, a.point.y));
        for (int i = 0; i < pointRanks.length; i++)
            pointRanks[i].rank += i;
        Arrays.sort(pointRanks, Comparator.comparingInt(a -> a.rank));
        Point bottomRight = pointRanks[0].point;
        return bottomRight;
    }
}
