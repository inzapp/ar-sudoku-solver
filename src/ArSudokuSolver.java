import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import org.opencv.videoio.VideoCapture;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

class SudokuAlgorithmSolver {
    static class Node {
        int x;
        int y;

        Node(int x, int y) {
            this.x = x;
            this.y = y;
        }
    }

    private boolean[][] checkCol = new boolean[9][10];
    private boolean[][] checkRow = new boolean[9][10];
    private boolean[][] checkBox = new boolean[9][10];

    private int[][] solve(int[][] sudoku, int cnt, ArrayList<Node> nodes, int idx) {
        if (cnt <= idx)
            return sudoku;
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

    int[][] getAnswer(int[][] sudoku) {
        int cnt = 0;
        ArrayList<Node> nodes = new ArrayList<>();

        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (sudoku[i][j] == 0) {
                    cnt++;
                    nodes.add(new Node(i, j));
                } else {
                    checkCol[i][sudoku[i][j]] = true;
                    checkRow[j][sudoku[i][j]] = true;
                    checkBox[(i / 3) * 3 + (j / 3)][sudoku[i][j]] = true;
                }
            }
        }

        return solve(sudoku, cnt, nodes, 0);
    }
}

public class ArSudokuSolver {
    private static boolean VIEW_PROGRESS = true;

    static {
        System.load("C:\\inz\\lib\\opencv_java411.dll");
    }

    static class PointRank {
        PointRank(Point point) {
            this.point = point;
        }

        int rank = 0;
        Point point;
    }

    public static void main(String[] args) {
        final int skipFrameCnt = 0;
        int cnt = 0;
        VideoCapture vc = new VideoCapture("C:\\inz\\sudoku.mp4");
        Mat frame = new Mat();
        while (vc.read(frame)) {
            if (cnt != skipFrameCnt) {
                ++cnt;
                continue;
            } else {
                cnt = 0;
            }
            process(frame, 1);
        }
    }

    private static void process(Mat raw, int fps) {
        //        int[][] answer = new SudokuAlgorithmSolver().getAnswer(new int[][]{
//                {0, 3, 0, 0, 5, 0, 0, 8, 0},
//                {9, 0, 0, 0, 0, 8, 0, 0, 6},
//                {0, 0, 0, 2, 0, 4, 0, 0, 0},
//                {0, 5, 6, 0, 0, 0, 1, 0, 0},
//                {7, 0, 0, 0, 0, 0, 0, 0, 2},
//                {0, 0, 9, 0, 0, 0, 3, 6, 0},
//                {0, 0, 0, 5, 0, 2, 0, 0, 0},
//                {4, 0, 0, 6, 0, 0, 0, 0, 8},
//                {0, 8, 0, 0, 1, 0, 0, 9, 0}
//        });
//
//        for (int[] row : answer) {
//            for (int col : row)
//                System.out.print(col + " ");
//            System.out.println();
//        }
//        System.exit(0);

        // load image
        Mat proc = raw.clone();

        // pre processing
        Imgproc.cvtColor(proc, proc, Imgproc.COLOR_BGR2GRAY);
        Imgproc.blur(proc, proc, new Size(2, 2));
        Imgproc.adaptiveThreshold(proc, proc, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 121, 15);
        Core.bitwise_not(proc, proc);

        if (VIEW_PROGRESS)
            HighGui.imshow("proc", proc.clone());

        // detect line
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
            if (VIEW_PROGRESS)
                Imgproc.line(raw, pt1, pt2, new Scalar(255, 0, 0), 2, Imgproc.LINE_AA, 0);
        }

        // find contours
        List<MatOfPoint> contourList = new ArrayList<>();
        Imgproc.findContours(proc, contourList, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);

        // calculate center of raw img
        Point rawCenter = new Point(raw.width() / 2, raw.height() / 2);

        // extract sudoku contour
        double maxArea = 0;
        MatOfPoint sudokuContour = new MatOfPoint();
        int rawResolution = raw.rows() * raw.cols();
        for (MatOfPoint contour : contourList) {
            Rect rect = Imgproc.boundingRect(contour);

            // filter : area
            if (!(maxArea < rect.area()))
                continue;

            // filter : area ratio
            if (rect.area() < rawResolution * 0.2 || rawResolution * 0.8 < rect.area())
                continue;

            // calculate center of contour using moment
            Moments moments = Imgproc.moments(contour, false);
            int x = (int) (moments.get_m10() / moments.get_m00());
            int y = (int) (moments.get_m01() / moments.get_m00());
            Point contourCenter = new Point(x, y);
            double distanceFromCenter = Core.norm(new MatOfPoint(rawCenter), new MatOfPoint(contourCenter));

            // filter : distance of contour center and frame center
            if (100 < distanceFromCenter)
                continue;

            // filter : h/w ratio
//            double ratio = rect.height / (double) rect.width;
//            if (ratio < 0.5 || 1.5 < ratio)
//                continue;


            // save
            sudokuContour = contour;
            maxArea = rect.area();
        }

        // calculate center of contour using moment
        Moments moments = Imgproc.moments(sudokuContour, false);
        int x = (int) (moments.get_m10() / moments.get_m00());
        int y = (int) (moments.get_m01() / moments.get_m00());
        Point contourCenter = new Point(x, y);

        if (VIEW_PROGRESS) {
            Imgproc.circle(raw, contourCenter, 10, new Scalar(0, 0, 255), -1);
            Imgproc.circle(raw, rawCenter, 10, new Scalar(255, 0, 0), -1);
        }

        // calculate convex hull of contour
        try {
            // convex hull
            MatOfInt hull = new MatOfInt();
            Imgproc.convexHull(sudokuContour, hull);

            // get hull idx list
            List<Integer> hullContourIdxList = hull.toList();
            Point[] contourArray = sudokuContour.toArray();
            Point[] hullPoints = new Point[hull.rows()];

            // copy hull point to array
            for (int i = 0; i < hullContourIdxList.size(); i++)
                hullPoints[i] = contourArray[hullContourIdxList.get(i)];

            // convert hull to list
            List<MatOfPoint> hullList = new ArrayList<>();
            hullList.add(new MatOfPoint(hullPoints));

            // draw hull
            if (VIEW_PROGRESS)
                Imgproc.drawContours(raw, hullList, 0, new Scalar(0, 255, 0), 2);
        } catch (Exception e) {
            // empty
        }

        // perspective transform with sudoku contour
        // copy points to point rank and sort
        try {
            Point[] points = sudokuContour.toArray();
            PointRank[] pointRanks = new PointRank[points.length];
            for (int i = 0; i < pointRanks.length; i++)
                pointRanks[i] = new PointRank(points[i]);

            Point topLeft;
            Arrays.sort(pointRanks, Comparator.comparingDouble(a -> a.point.y));
            for (int i = 0; i < pointRanks.length; i++)
                pointRanks[i].rank += i;
            Arrays.sort(pointRanks, Comparator.comparingDouble(a -> a.point.x));
            for (int i = 0; i < pointRanks.length; i++)
                pointRanks[i].rank += i;
            Arrays.sort(pointRanks, Comparator.comparingInt(a -> a.rank));
            topLeft = pointRanks[0].point;

            //calculate top right point
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

            //calculate bottom left point
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

            //calculate bottom right point
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

            if (VIEW_PROGRESS) {
                Imgproc.circle(raw, topLeft, 10, new Scalar(0, 0, 255), 2);
                Imgproc.circle(raw, topRight, 10, new Scalar(0, 0, 255), 2);
                Imgproc.circle(raw, bottomLeft, 10, new Scalar(0, 0, 255), 2);
                Imgproc.circle(raw, bottomRight, 10, new Scalar(0, 0, 255), 2);
            }

            // perspective transform with sudoku contour
            Mat before = new MatOfPoint2f(topLeft, topRight, bottomLeft, bottomRight);
            Mat after = new MatOfPoint2f(new Point(0, 0), new Point(proc.cols(), 0),
                    new Point(0, proc.rows()), new Point(proc.cols(), proc.rows()));
            Mat perspectiveTransformer = Imgproc.getPerspectiveTransform(before, after);
            Imgproc.warpPerspective(proc, proc, perspectiveTransformer, proc.size());
            Imgproc.resize(proc, proc, new Size(28 * 9, 28 * 9));
            if (VIEW_PROGRESS)
                HighGui.imshow("transform", proc);
        } catch (Exception e) {
            // empty
        }

        // split mat into 9 * 9
//        Mat[] elements = new Mat[9 * 9];
//        int offset = 28;
//        int index = 0;
//        for (int i = 0; i < 9 * offset; i += offset) {
//            for (int j = 0; j < 9 * offset; j += offset)
//                elements[index++] = proc.rowRange(i, i + offset).colRange(j, j + offset);
//        }

//        for (Mat cur : elements) {
//            HighGui.imshow("res", cur);
//            HighGui.waitKey(fps);
//        }

        HighGui.imshow("cam", raw);
        HighGui.waitKey(fps);
//        try {
//            Thread.sleep(300);
//        } catch (InterruptedException e) {
//            e.printStackTrace();
//        }
    }
}
