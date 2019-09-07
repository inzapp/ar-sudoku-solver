package com.inzapp.arSudokuSolver.core;

import com.inzapp.arSudokuSolver.config.Config;
import com.inzapp.arSudokuSolver.util.View;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class SudokuAnswerRenderer {
    /**
     *
     * @param raw
     * @param perspectiveTransformer
     * @param unsolvedSudoku
     * @param solvedSudoku
     * @param inverseTransformer
     * @param pureResult
     */
    public void renderAnswer(Mat raw, Mat perspectiveTransformer, int[][] unsolvedSudoku, int[][] solvedSudoku, Mat inverseTransformer, Mat pureResult) {
        Mat perspective = new Mat();
        Imgproc.warpPerspective(raw, perspective, perspectiveTransformer, raw.size());

        // resize 1 : 1 rectangle shape
        int smaller = Math.min(raw.rows(), raw.cols());
        Imgproc.resize(perspective, perspective, new Size(smaller, smaller));

        renderToPerspective(perspective, unsolvedSudoku, solvedSudoku);
        Imgproc.resize(perspective, perspective, raw.size());

        // get inverse transform
        Mat originalPerspective = new Mat();
        Imgproc.warpPerspective(perspective, originalPerspective, inverseTransformer, raw.size());

        if (Config.VIEW_PROGRESS)
            View.sudokuArea = originalPerspective.clone();

        overlay(raw, originalPerspective, pureResult);
    }

    /**
     * render perspective to answer text
     *
     * @param perspective
     * @param unsolvedSudoku
     * @param solvedSudoku
     */
    private void renderToPerspective(Mat perspective, int[][] unsolvedSudoku, int[][] solvedSudoku) {
        int rowOffset = perspective.rows() / 9;
        int colOffset = perspective.cols() / 9;
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                if (unsolvedSudoku[i][j] == 0)
                    Imgproc.putText(perspective,
                            String.valueOf(solvedSudoku[i][j]),
                            new Point(j * colOffset + 16, i * rowOffset + 39),
                            Imgproc.FONT_HERSHEY_SIMPLEX, 1.3, new Scalar(0, 0, 255), 3);
            }
        }
    }

    /**
     *
     * @param raw
     * @param originalPerspective
     * @param pureResult
     */
    private void overlay(Mat raw, Mat originalPerspective, Mat pureResult) {
        for (int row = 0; row < raw.rows(); ++row) {
            for (int col = 0; col < raw.cols(); ++col) {
                if (originalPerspective.get(row, col)[0] == 0 &&
                        originalPerspective.get(row, col)[1] == 0 &&
                        originalPerspective.get(row, col)[2] == 0)
                    continue;

                pureResult.put(row, col, originalPerspective.get(row, col));
            }
        }
    }
}
