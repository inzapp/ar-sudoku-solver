package com.inzapp.arSudokuSolver.core;

import com.inzapp.arSudokuSolver.config.Config;
import com.inzapp.arSudokuSolver.util.View;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.ANN_MLP;
import org.opencv.ml.StatModel;

public class SudokuArrayConverter {
    private StatModel model;

    /**
     *
     */
    public SudokuArrayConverter() {
        model = ANN_MLP.load("model.xml");
    }

    /**
     * @param perspectiveTransformer
     * @param proc
     * @return
     */
    public int[][] convert(Mat perspectiveTransformer, Mat proc) {
        // perspective transform with processing sudoku contour
        Imgproc.warpPerspective(proc, proc, perspectiveTransformer, proc.size());

        Mat[][] elements = this.split(proc, 28);
        Mat res = new Mat();
        int[][] unsolvedSudoku = new int[9][9];
        for (int i = 0; i < elements.length; ++i) {
            for (int j = 0; j < elements[i].length; ++j) {
                Mat cur = elements[i][j];
                cur.convertTo(cur, CvType.CV_32FC1, 1 / 255.0f);
                cur = cur.reshape(cur.channels(), 1);
                unsolvedSudoku[i][j] = (int) model.predict(cur, res);
            }
        }

        // print unsolved sudoku
        if (Config.VIEW_PROGRESS) {
            for (int[] row : unsolvedSudoku) {
                for (int cur : row)
                    System.out.print(cur + " ");
                System.out.println();
            }
            System.out.println();
        }

        return unsolvedSudoku;
    }

    /**
     * calculate perspective transformer
     *
     * @param corners
     * @param proc
     * @return
     */
    public Mat[] getPerspectiveTransformers(Point[] corners, Mat proc) {
        try {
            Mat before = new MatOfPoint2f(corners[0], corners[1], corners[2], corners[3]);
            Mat after = new MatOfPoint2f(new Point(0, 0), new Point(proc.cols(), 0),
                    new Point(0, proc.rows()), new Point(proc.cols(), proc.rows()));
            Mat perspectiveTransformer = Imgproc.getPerspectiveTransform(before, after);
            Mat inverseTransformer = Imgproc.getPerspectiveTransform(after, before);
            return new Mat[]{perspectiveTransformer, inverseTransformer};
        } catch (Exception e) {
            return null;
        }
    }

    /**
     * @param proc
     * @param splitSize
     * @return
     */
    private Mat[][] split(Mat proc, int splitSize) {
        // resize to (28 * 9) * (28 * 9) : 28 is column of train data
        if (Config.VIEW_PROGRESS) {
            Mat transform = new Mat();
            Imgproc.resize(proc, transform, new Size(splitSize * 15, splitSize * 15));
            View.transform = transform;
        }

        // split mat into 9 * 9
        Imgproc.resize(proc, proc, new Size(splitSize * 9, splitSize * 9));
        Mat[][] elements = new Mat[9][9];
        int offset = splitSize;
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j)
                elements[i][j] = proc.
                        rowRange(i * offset, i * offset + offset).
                        colRange(j * offset, j * offset + offset);
        }

        if (Config.VIEW_PROGRESS) {
            for (int i = 0; i < elements.length; ++i) {
                for (int j = 0; j < elements[i].length; ++j) {
                    Imgproc.resize(elements[i][j], View.elements[i][j], new Size(splitSize * 15 / 9, splitSize * 15 / 9));
                }
            }
        }
        return elements;
    }
}