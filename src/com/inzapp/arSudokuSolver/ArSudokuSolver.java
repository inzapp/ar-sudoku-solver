package com.inzapp.arSudokuSolver;

import com.inzapp.arSudokuSolver.config.Config;
import com.inzapp.arSudokuSolver.core.*;
import com.inzapp.arSudokuSolver.util.View;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import org.opencv.ml.ANN_MLP;
import org.opencv.ml.StatModel;
import org.opencv.videoio.VideoCapture;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class ArSudokuSolver {
    private SudokuContourFinder sudokuContourFinder;
    private SudokuArrayConverter sudokuArrayConverter;
    private ConvexHullToContourConverter convexHullToContourConverter;
    private SudokuCornerExtractor sudokuCornerExtractor;
    private AlgorithmSolver sudokuAlgorithmSolver;
    private ExecutorService executorService;

    ArSudokuSolver() {
        sudokuContourFinder = new SudokuContourFinder();
        sudokuArrayConverter = new SudokuArrayConverter();
        convexHullToContourConverter = new ConvexHullToContourConverter();
        sudokuCornerExtractor = new SudokuCornerExtractor();
        sudokuAlgorithmSolver = new AlgorithmSolver();
        executorService = Executors.newSingleThreadExecutor();
    }

    static {
        System.load("C:\\inz\\lib\\opencv_java411.dll");
    }

    public static void main(String[] args) {
        final int skipFrameCnt = 7;
        int cnt = 0;
        VideoCapture vc = new VideoCapture("C:\\inz\\numgigi.mp4");
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

    private Mat render(Mat raw) {
        Mat pureResult = raw.clone();
        Mat progress = raw.clone();
        Mat proc = raw.clone();
        try {
            preProcess(proc);

            drawLine(progress, proc);

            MatOfPoint sudokuContour = sudokuContourFinder.find(progress, proc);

            MatOfPoint hullPoints = convexHullToContourConverter.convert(progress, sudokuContour);

            Point[] corners = sudokuCornerExtractor.extract(sudokuContour, progress);

            Mat[] perspectiveTransformers = sudokuArrayConverter.getPerspectiveTransformers(corners, proc);

            Mat perspectiveTransformer = perspectiveTransformers[0];

            Mat inverseTransformer = perspectiveTransformers[1];

            int[][] unsolvedSudoku = sudokuArrayConverter.convert(perspectiveTransformer, proc);

            // calculate sudoku answer
            Callable<int[][]> callable = () -> sudokuAlgorithmSolver.getAnswer2d(unsolvedSudoku);
            int[][] solvedSudoku = executorService.submit(callable).get(100, TimeUnit.MILLISECONDS);// get perspective transformation of raw sudoku area
            Mat perspective = new Mat();
            Imgproc.warpPerspective(raw, perspective, perspectiveTransformer, raw.size());

            // resize 1 : 1 rectangle shape
            int smaller = Math.min(raw.rows(), raw.cols());
            Imgproc.resize(perspective, perspective, new Size(smaller, smaller));

            // render perspective to answer text
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

            Imgproc.resize(perspective, perspective, raw.size());

            // get inverse transform
            Mat originalPerspective = new Mat();
            Imgproc.warpPerspective(perspective, originalPerspective, inverseTransformer, raw.size());

            if (Config.VIEW_PROGRESS) {
                View.sudokuArea = originalPerspective.clone();
            }

            // overlay
            for (int row = 0; row < raw.rows(); ++row) {
                for (int col = 0; col < raw.cols(); ++col) {
                    if (originalPerspective.get(row, col)[0] == 0 &&
                            originalPerspective.get(row, col)[1] == 0 &&
                            originalPerspective.get(row, col)[2] == 0)
                        continue;

                    pureResult.put(row, col, originalPerspective.get(row, col));
                }
            }
        } catch (Exception e) {
        }

        View.pureResult = pureResult;
        View.progress = progress;
        return raw;
    }

    private void preProcess(Mat proc) {
        // pre processing
        Imgproc.cvtColor(proc, proc, Imgproc.COLOR_BGR2GRAY);
        Imgproc.blur(proc, proc, new Size(2, 2));
        Imgproc.adaptiveThreshold(proc, proc, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 121, 15);
        Core.bitwise_not(proc, proc);
    }

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
