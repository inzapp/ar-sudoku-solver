import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.*;

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
    static {
        System.load("C:\\inzapp\\lib\\opencv\\opencv_java410.dll");
    }

    static class PointRank {
        PointRank(Point point) {
            this.point = point;
        }

        int rank = 0;
        Point point;
    }

    public static void main(String[] args) {
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
        Mat raw = Imgcodecs.imread("C:\\inzapp\\sudoku\\s13.jpg", Imgcodecs.IMREAD_ANYCOLOR);
        Mat proc = raw.clone();

        // pre processing
        Imgproc.cvtColor(proc, proc, Imgproc.COLOR_BGR2GRAY);
        Imgproc.blur(proc, proc, new Size(2, 2));
        Imgproc.adaptiveThreshold(proc, proc, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 201, 7);
        Core.bitwise_not(proc, proc);

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
            Imgproc.line(raw, pt1, pt2, new Scalar(255, 0, 0), 2, Imgproc.LINE_AA, 0);
        }

        HighGui.imshow("img", raw);
        HighGui.waitKey(0);

        // detect sudoku contour
        // find contours
        List<MatOfPoint> contourList = new ArrayList<>();
        Imgproc.findContours(proc, contourList, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

        // find biggest contour
        double maxArea = 0;
        Rect biggestRect = new Rect();
        MatOfPoint biggestContour = new MatOfPoint();
        for (MatOfPoint contour : contourList) {
            Rect rect = Imgproc.boundingRect(contour);
            if (maxArea < rect.area()) {
                biggestContour = contour;
                biggestRect = rect;
                maxArea = rect.area();
            }
        }
        Imgproc.drawContours(raw, Collections.singletonList(biggestContour), 0, new Scalar(0, 255, 0), 2);
//        Imgproc.rectangle(raw, biggestRect, new Scalar(0, 0, 255), 2);
        for (Point cur : biggestContour.toArray())
            System.out.println(cur);

        // perspective transform with sudoku contour
        // copy points to point rank and sort
        Point[] points = biggestContour.toArray();
        PointRank[] pointRanks = new PointRank[points.length];
        for (int i = 0; i < pointRanks.length; i++)
            pointRanks[i] = new PointRank(points[i]);

        // get 4 corner point of sudoku contour
        // calculate top left point
        Point topLeft;
        Arrays.sort(pointRanks, Comparator.comparingDouble(a -> a.point.x));
        for (int i = 0; i < pointRanks.length; i++)
            pointRanks[i].rank += i + 1;
        Arrays.sort(pointRanks, Comparator.comparingDouble(a -> a.point.y));
        for (int i = 0; i < pointRanks.length; i++)
            pointRanks[i].rank += i + 1;
        Arrays.sort(pointRanks, Comparator.comparingInt(a -> a.rank));
        topLeft = pointRanks[0].point;

        //calculate top right point
        Point topRight;
        for (int i = 0; i < pointRanks.length; i++)
            pointRanks[i].rank = 0;
        Arrays.sort(pointRanks, (a, b) -> Double.compare(b.point.x, a.point.x));
        for (int i = 0; i < pointRanks.length; i++)
            pointRanks[i].rank += i + 1;
        Arrays.sort(pointRanks, Comparator.comparingDouble(a -> a.point.y));
        for (int i = 0; i < pointRanks.length; i++)
            pointRanks[i].rank += i + 1;
        Arrays.sort(pointRanks, Comparator.comparingInt(a -> a.rank));
        topRight = pointRanks[0].point;

        //calculate top right point
        Point bottomLeft;
        for (int i = 0; i < pointRanks.length; i++)
            pointRanks[i].rank = 0;
        Arrays.sort(pointRanks, Comparator.comparingDouble(a -> a.point.x));
        for (int i = 0; i < pointRanks.length; i++)
            pointRanks[i].rank += i + 1;
        Arrays.sort(pointRanks, (a, b) -> Double.compare(b.point.y, a.point.y));
        for (int i = 0; i < pointRanks.length; i++)
            pointRanks[i].rank += i + 1;
        Arrays.sort(pointRanks, Comparator.comparingInt(a -> a.rank));
        bottomLeft = pointRanks[0].point;

        //calculate bottom right point
        Point bottomRight;
        for (int i = 0; i < pointRanks.length; i++)
            pointRanks[i].rank = 0;
        Arrays.sort(pointRanks, (a, b) -> Double.compare(b.point.x, a.point.x));
        for (int i = 0; i < pointRanks.length; i++)
            pointRanks[i].rank += i + 1;
        Arrays.sort(pointRanks, (a, b) -> Double.compare(b.point.y, a.point.y));
        for (int i = 0; i < pointRanks.length; i++)
            pointRanks[i].rank += i + 1;
        Arrays.sort(pointRanks, Comparator.comparingInt(a -> a.rank));
        bottomRight = pointRanks[0].point;

        Imgproc.circle(raw, topLeft, 10, new Scalar(0, 0, 255), 2);
        Imgproc.circle(raw, topRight, 10, new Scalar(0, 0, 255), 2);
        Imgproc.circle(raw, bottomLeft, 10, new Scalar(0, 0, 255), 2);
        Imgproc.circle(raw, bottomRight, 10, new Scalar(0, 0, 255), 2);

        HighGui.imshow("res", raw);
        HighGui.waitKey(0);

        // perspective transform with sudoku contour
        Mat before = new MatOfPoint2f(topLeft, topRight, bottomLeft, bottomRight);
        Mat after = new MatOfPoint2f(new Point(0, 0), new Point(proc.cols(), 0),
                new Point(0, proc.rows()), new Point(proc.cols(), proc.rows()));
        Mat perspectiveTransformer = Imgproc.getPerspectiveTransform(before, after);
        Imgproc.warpPerspective(proc, proc, perspectiveTransformer, proc.size());
        Imgproc.resize(proc, proc, new Size(28 * 9, 28 * 9));

        // split mat into 9 * 9
        Mat[] elements = new Mat[9 * 9];
        int offset = proc.rows() / 9;
        int index = 0;
        for (int i = 0; i < 9 * offset; i += offset) {
            for (int j = 0; j < 9 * offset; j += offset)
                elements[index++] = proc.rowRange(i, i + offset).colRange(j, j + offset);
        }

        HighGui.imshow("res", proc);
        HighGui.waitKey(0);

        for (Mat cur : elements) {
            HighGui.imshow("res", cur);
            HighGui.waitKey(0);
        }

        HighGui.destroyAllWindows();
    }
}
