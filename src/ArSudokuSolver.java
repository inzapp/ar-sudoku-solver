import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.*;

class SudokuAlgorithmSolver {
    private class Cell {
        int val;
        List<Integer> candidate;

        Cell(int val) {
            if (val == 0) {
                this.val = 0;
                candidate = new ArrayList<>(Arrays.asList(
                        1, 2, 3, 4, 5, 6, 7, 8, 9
                ));
            } else {
                this.val = val;
                candidate = new ArrayList<>();
            }
        }

        void remoteCandidate(int val) {
            candidate.remove(new Integer(val));
            if (candidate.size() == 1)
                this.val = candidate.get(0);
        }
    }

    public void solve() {
        Scanner sc = new Scanner(System.in);
        Cell[][][] pan = new Cell[9][3][3];
        for (int sq = 0; sq < 9; ++sq) {
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    pan[sq][i][j] = new Cell(sc.nextInt());
                }
            }
        }

        pan[0][0][0].remoteCandidate(1);
        pan[0][0][0].remoteCandidate(2);
        pan[0][0][0].remoteCandidate(3);
        pan[0][0][0].remoteCandidate(4);
        pan[0][0][0].remoteCandidate(5);
        pan[0][0][0].remoteCandidate(6);
        pan[0][0][0].remoteCandidate(7);
        System.out.println(pan[0][0][0].val);
        pan[0][0][0].remoteCandidate(8);

        for (int cur : pan[0][0][0].candidate)
            System.out.print(cur + " ");

        System.out.println(pan[0][0][0].val);
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
        new SudokuAlgorithmSolver().solve();
        System.exit(0);

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
        for (int x = 0; x < lines.rows(); x++) {
            double rho = lines.get(x, 0)[0];
            double theta = lines.get(x, 0)[1];
            double a = Math.cos(theta);
            double b = Math.sin(theta);
            double x0 = a * rho;
            double y0 = b * rho;
            Point pt1 = new Point(Math.round(x0 + 1000 * (-b)), Math.round(y0 + 1000 * (a)));
            Point pt2 = new Point(Math.round(x0 - 1000 * (-b)), Math.round(y0 - 1000 * (a)));
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
