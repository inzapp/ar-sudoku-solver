package com.inzapp.arSudokuSolver;

import com.inzapp.arSudokuSolver.config.Config;
import com.inzapp.arSudokuSolver.util.View;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import org.opencv.ml.ANN_MLP;
import org.opencv.ml.StatModel;
import org.opencv.videoio.VideoCapture;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;


class SudokuAlgorithmSolver {
    private static class Node {
        int x;
        int y;

        Node(int x, int y) {
            this.x = x;
            this.y = y;
        }
    }

    private boolean[][] checkCol;
    private boolean[][] checkRow;
    private boolean[][] checkBox;

    private int[][] solve(int[][] sudoku, int cnt, List<Node> nodes, int idx) {
        if (cnt <= idx) {
            if (Config.VIEW_PROGRESS) {
                for (int[] row : sudoku) {
                    for (int cur : row)
                        System.out.print(cur + " ");
                    System.out.println();
                }
            }
            return sudoku;
        }
        Node node = nodes.get(idx);

        // brute force 1 ~ 9
        for (int i = 1; i <= 9; ++i) {
            if (checkCol[node.x][i])
                continue;
            if (checkRow[node.y][i])
                continue;
            if (checkBox[(node.x / 3) * 3 + (node.y) / 3][i])
                continue;

            checkCol[node.x][i] = true;
            checkRow[node.y][i] = true;
            checkBox[(node.x / 3) * 3 + (node.y) / 3][i] = true;
            sudoku[node.x][node.y] = i;
            if (solve(sudoku, cnt, nodes, idx + 1) != null)
                return sudoku;

            // back tracking
            sudoku[node.x][node.y] = 0;
            checkCol[node.x][i] = false;
            checkRow[node.y][i] = false;
            checkBox[(node.x / 3) * 3 + (node.y) / 3][i] = false;
        }

        return null;
    }

    int[][] getAnswer2d(int[][] sudoku) {
        int cnt = 0;
        List<Node> nodes = new ArrayList<>();
        this.checkBox = new boolean[9][10];
        this.checkCol = new boolean[9][10];
        this.checkRow = new boolean[9][10];

        int[][] unsolvedSudoku = new int[9][9];
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j)
                unsolvedSudoku[i][j] = sudoku[i][j];
        }

        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                if (sudoku[i][j] == 0) {
                    ++cnt;
                    nodes.add(new Node(i, j));
                } else {
                    checkCol[i][unsolvedSudoku[i][j]] = true;
                    checkRow[j][unsolvedSudoku[i][j]] = true;
                    checkBox[(i / 3) * 3 + (j / 3)][unsolvedSudoku[i][j]] = true;
                }
            }
        }

        return solve(unsolvedSudoku, cnt, nodes, 0);
    }
}

class SudokuCornerExtractor {
    // used for calculating 4 corner of sudoku contour
    private static class PointRank {
        PointRank(Point point) {
            this.point = point;
        }

        int rank = 0;
        Point point;
    }

    Point[] extract(MatOfPoint sudokuContour, Mat progress) {
        // find 4 corner of sudoku contour to perspective transform
        // copy points to point rank and sort
        try {
            Point[] points = sudokuContour.toArray();
            PointRank[] pointRanks = new PointRank[points.length];
            for (int i = 0; i < pointRanks.length; i++)
                pointRanks[i] = new PointRank(points[i]);

            // top left
            Point topLeft;
            Arrays.sort(pointRanks, Comparator.comparingDouble(a -> a.point.y));
            for (int i = 0; i < pointRanks.length; i++)
                pointRanks[i].rank += i;
            Arrays.sort(pointRanks, Comparator.comparingDouble(a -> a.point.x));
            for (int i = 0; i < pointRanks.length; i++)
                pointRanks[i].rank += i;
            Arrays.sort(pointRanks, Comparator.comparingInt(a -> a.rank));
            topLeft = pointRanks[0].point;

            // top right
            Point topRight;
            for (int i = 0; i < pointRanks.length; i++)
                pointRanks[i].rank = 0;
            Arrays.sort(pointRanks, (a, b) -> Double.compare(b.point.x, a.point.x));
            for (int i = 0; i < pointRanks.length; i++)
                pointRanks[i].rank += i;
            Arrays.sort(pointRanks, Comparator.comparingDouble(a -> a.point.y));
            for (int i = 0; i < pointRanks.length; i++)
                pointRanks[i].rank += i;
            Arrays.sort(pointRanks, Comparator.comparingInt(a -> a.rank));
            topRight = pointRanks[0].point;

            // bottom left
            Point bottomLeft;
            for (int i = 0; i < pointRanks.length; i++)
                pointRanks[i].rank = 0;
            Arrays.sort(pointRanks, Comparator.comparingDouble(a -> a.point.x));
            for (int i = 0; i < pointRanks.length; i++)
                pointRanks[i].rank += i;
            Arrays.sort(pointRanks, (a, b) -> Double.compare(b.point.y, a.point.y));
            for (int i = 0; i < pointRanks.length; i++)
                pointRanks[i].rank += i;
            Arrays.sort(pointRanks, Comparator.comparingInt(a -> a.rank));
            bottomLeft = pointRanks[0].point;

            // bottom right
            Point bottomRight;
            for (int i = 0; i < pointRanks.length; i++)
                pointRanks[i].rank = 0;
            Arrays.sort(pointRanks, (a, b) -> Double.compare(b.point.x, a.point.x));
            for (int i = 0; i < pointRanks.length; i++)
                pointRanks[i].rank += i;
            Arrays.sort(pointRanks, (a, b) -> Double.compare(b.point.y, a.point.y));
            for (int i = 0; i < pointRanks.length; i++)
                pointRanks[i].rank += i;
            Arrays.sort(pointRanks, Comparator.comparingInt(a -> a.rank));
            bottomRight = pointRanks[0].point;

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
}

class SudokuContourFinder {
    MatOfPoint find(Mat progress, Mat proc) {
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

            // filter : h/w ratio
//            double ratio = rect.height / (double) rect.width;
//            if (ratio < 0.5 || 1.5 < ratio)
//                continue;

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

    private double getDistance(Point a, Point b) {
        return Core.norm(new MatOfPoint(a), new MatOfPoint(b));
    }

    private Point getCenterPoint(MatOfPoint sudokuContour) {
        // calculate center of sudoku contour using moment
        Moments moments = Imgproc.moments(sudokuContour, false);
        int x = (int) (moments.get_m10() / moments.get_m00());
        int y = (int) (moments.get_m01() / moments.get_m00());
        return new Point(x, y);
    }
}

class ConvexHullToContourConverter {
    MatOfPoint convert(Mat progress, MatOfPoint sudokuContour) {
        if(!Config.VIEW_PROGRESS)
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

class SudokuArrayConverter {
    private StatModel model;

    SudokuArrayConverter() {
        model = ANN_MLP.load("model.xml");
    }

    int[][] convert(Mat perspectiveTransformer, Mat proc) {
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

    Mat[] getPerspectiveTransformers(Point[] corners, Mat proc) {
        // calculate perspective transformer
        Mat before = new MatOfPoint2f(corners[0], corners[1], corners[2], corners[3]);
        Mat after = new MatOfPoint2f(new Point(0, 0), new Point(proc.cols(), 0),
                new Point(0, proc.rows()), new Point(proc.cols(), proc.rows()));
        Mat perspectiveTransformer = Imgproc.getPerspectiveTransform(before, after);
        Mat inverseTransformer = Imgproc.getPerspectiveTransform(after, before);
        return new Mat[]{perspectiveTransformer, inverseTransformer};
    }

    private Mat[][] split(Mat proc, int splitSize) {
        // resize to (28 * 9) * (28 * 9) : 28 is column of train data
        if (Config.VIEW_PROGRESS) {
            Mat transform = new Mat();
            Imgproc.resize(proc, transform, new Size(splitSize * 15, splitSize * 15));
            View.transform = transform;
        }

        Imgproc.resize(proc, proc, new Size(splitSize * 9, splitSize * 9));
        // split mat into 9 * 9
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
                    Mat elementView = new Mat();
                    Imgproc.resize(elements[i][j], View.elements[i][j], new Size(splitSize * 15 / 9, splitSize * 15 / 9));
                }
            }
        }
        return elements;
    }
}

public class ArSudokuSolver {
    private SudokuContourFinder sudokuContourFinder;
    private SudokuArrayConverter sudokuArrayConverter;
    private ConvexHullToContourConverter convexHullToContourConverter;
    private SudokuCornerExtractor sudokuCornerExtractor;
    private SudokuAlgorithmSolver sudokuAlgorithmSolver;
    private ExecutorService executorService;

    ArSudokuSolver() {
        sudokuContourFinder = new SudokuContourFinder();
        sudokuArrayConverter = new SudokuArrayConverter();
        convexHullToContourConverter = new ConvexHullToContourConverter();
        sudokuCornerExtractor = new SudokuCornerExtractor();
        sudokuAlgorithmSolver = new SudokuAlgorithmSolver();
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
