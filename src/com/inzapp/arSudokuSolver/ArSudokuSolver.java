package com.inzapp.arSudokuSolver;

import com.inzapp.arSudokuSolver.config.Config;
import com.inzapp.arSudokuSolver.core.*;
import com.inzapp.arSudokuSolver.util.View;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

public class ArSudokuSolver {
    private SudokuContourFinder sudokuContourFinder;
    private SudokuArrayConverter sudokuArrayConverter;
    private ConvexHullToContourConverter convexHullToContourConverter;
    private SudokuCornerExtractor sudokuCornerExtractor;
    private SudokuAlgorithmSolver sudokuAlgorithmSolver;
    private SudokuAnswerRenderer sudokuAnswerRenderer;

    /**
     * default constructor
     */
    private ArSudokuSolver() {
        this.sudokuContourFinder = new SudokuContourFinder();
        this.sudokuArrayConverter = new SudokuArrayConverter();
        this.convexHullToContourConverter = new ConvexHullToContourConverter();
        this.sudokuCornerExtractor = new SudokuCornerExtractor();
        this.sudokuAlgorithmSolver = new SudokuAlgorithmSolver();
        this.sudokuAnswerRenderer = new SudokuAnswerRenderer();
    }

    /**
     * entry point
     *
     * @param args not used
     */
    public static void main(String[] args) {
        final int skipFrameCnt = 7;
        int cnt = 0;
        VideoCapture vc = new VideoCapture("sample.mp4");
        ArSudokuSolver solver = new ArSudokuSolver();
        Mat frame = new Mat();
        View.init();
        while (vc.read(frame)) {
            if (cnt != skipFrameCnt) {
                ++cnt;
                continue;
            } else {
                cnt = 0;
            }

            solver.render(frame);
            View.show(1);
        }

        System.out.println("end");
        System.exit(0);
    }

    /**
     * get frame from sudoku video and render answer
     *
     * @param raw each frame of video
     */
    private void render(Mat raw) {
        Mat pureResult = raw.clone();
        Mat progress = raw.clone();
        Mat proc = raw.clone();
        preProcess(proc);
        drawLine(progress, proc);
        MatOfPoint sudokuContour = this.sudokuContourFinder.find(progress, proc);
        MatOfPoint hullPoints = this.convexHullToContourConverter.convert(progress, sudokuContour);
        Point[] corners = this.sudokuCornerExtractor.extract(sudokuContour, progress);

        View.pureResult = pureResult;
        Mat[] perspectiveTransformers = this.sudokuArrayConverter.getPerspectiveTransformers(corners, proc);
        if (perspectiveTransformers == null)
            return;

        Mat perspectiveTransformer = perspectiveTransformers[0];
        Mat inverseTransformer = perspectiveTransformers[1];
        int[][] unsolvedSudoku = this.sudokuArrayConverter.convert(perspectiveTransformer, proc);
        int[][] solvedSudoku = this.sudokuAlgorithmSolver.solveInTime(unsolvedSudoku, 100);
        if (solvedSudoku == null)
            return;

        this.sudokuAnswerRenderer.renderAnswer(raw, perspectiveTransformer, unsolvedSudoku, solvedSudoku, inverseTransformer, pureResult);
        View.pureResult = pureResult;
        View.progress = progress;
    }

    /**
     * pre processing of processing
     *
     * @param proc mat for processing
     */
    private void preProcess(Mat proc) {
        // pre processing
        Imgproc.cvtColor(proc, proc, Imgproc.COLOR_BGR2GRAY);
        Imgproc.blur(proc, proc, new Size(2, 2));
        Imgproc.adaptiveThreshold(proc, proc, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 121, 15);
        Core.bitwise_not(proc, proc);
    }

    /**
     * draw line of hough transformed frame
     *
     * @param progress mat for viewing progress
     * @param proc     mat for processing
     */
    private void drawLine(Mat progress, Mat proc) {
        // detect line
        if (Config.VIEW_PROGRESS) {
            Mat canny = proc.clone();
            Imgproc.Canny(canny, canny, 100, 100);
            Mat lines = new Mat();
            Imgproc.HoughLines(canny, lines, 1, Math.PI / 180, 150);
            for (int row = 0; row < lines.rows(); row++) {
                double rho = lines.get(row, 0)[0];
                double theta = lines.get(row, 0)[1];
                double a = Math.cos(theta);
                double b = Math.sin(theta);
                double x = a * rho;
                double y = b * rho;
                Point pt1 = new Point(Math.round(x + 1000 * (-b)), Math.round(y + 1000 * (a)));
                Point pt2 = new Point(Math.round(x - 1000 * (-b)), Math.round(y - 1000 * (a)));
                Imgproc.line(progress, pt1, pt2, new Scalar(255, 0, 0), 2, Imgproc.LINE_AA, 0);
            }
        }
    }
}
