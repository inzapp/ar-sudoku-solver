import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import org.opencv.ml.ANN_MLP;
import org.opencv.videoio.VideoCapture;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

class SudokuAlgorithmSolver {
    private static class Node {
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

    private int[][] getAnswer2d(int[][] sudoku) {
        int cnt = 0;
        ArrayList<Node> nodes = new ArrayList<>();

        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                if (sudoku[i][j] == 0) {
                    ++cnt;
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

    int[] getAnswer(int[] unsolved1d) {
        int idx = 0;
        int[][] unsolved2d = new int[9][9];
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                unsolved2d[i][j] = unsolved1d[idx++];
            }
        }

        int[][] answer2d = getAnswer2d(unsolved2d);
        if (answer2d == null)
            return null;

        int[] answer1d = new int[81];
        idx = 0;
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
//                answer1d[idx] = unsolved1d[idx] == 0 ? answer2d[i][j] : 0;
                answer1d[idx] = answer2d[i][j];
                ++idx;
            }
        }

        return answer1d;
    }
}

public class ArSudokuSolver {
    private static boolean VIEW_PROGRESS = false;

    static {
        System.load("C:\\inz\\lib\\opencv_java411.dll");
    }

    // used for calculating 4 corner of sudoku contour
    private static class PointRank {
        PointRank(Point point) {
            this.point = point;
        }

        int rank = 0;
        Point point;
    }

    public static void main(String[] args) {
        // train with video name and sudoku arr
        final int skipFrameCnt = 1;
        int cnt = 0;
//        VideoCapture vc = new VideoCapture("C:\\inz\\lib\\sudoku_train_video_answer" + "\\" + videoName);
        VideoCapture vc = new VideoCapture("C:\\inz\\sudoku.mp4");
        Mat frame = new Mat();
        while (vc.read(frame)) {
            if (cnt != skipFrameCnt) {
                ++cnt;
                continue;
            } else {
                cnt = 0;
            }
            process(frame, 1, new int[][]{});
        }

        System.exit(0);

//        String dirPath = "C:\\inz\\lib\\sudoku_train_video_answer";
//        File[] files = new File("C:\\inz\\lib\\sudoku_train_video_answer").listFiles();
//        for (File cur : files) {
//            String[] splits = cur.getName().split("\\.");
//            if (splits[1].equals("txt")) {
//                try {
//
//                    // get sudoku arr
//                    Scanner sc = new Scanner(new FileInputStream(cur.getAbsolutePath()));
//                    int[][] sudoku = new int[9][9];
//                    for (int i = 0; i < sudoku.length; ++i) {
//                        for (int j = 0; j < sudoku[i].length; ++j) {
//                            sudoku[i][j] = sc.nextInt();
//                        }
//                    }
//
//                    // get video name
//                    String videoName = splits[0] + ".mp4";
//                    System.out.println(videoName);
//
////                    // train with video name and sudoku arr
////                    final int skipFrameCnt = 1;
////                    int cnt = 0;
////                    VideoCapture vc = new VideoCapture("C:\\inz\\lib\\sudoku_train_video_answer" + "\\" + videoName);
//////                    VideoCapture vc = new VideoCapture("C:\\inz\\sudoku.mp4");
////                    Mat frame = new Mat();
////                    while (vc.read(frame)) {
////                        if (cnt != skipFrameCnt) {
////                            ++cnt;
////                            continue;
////                        } else {
////                            cnt = 0;
////                        }
////                        process(frame, 1, sudoku);
////                    }
//
//                    System.out.println();
//                } catch (Exception e) {
//                    e.printStackTrace();
//                }
//            }
//        }
    }

    private static void process(Mat raw, int fps, int[][] sudoku) {
        // load image
        Mat proc = raw.clone();

        // pre processing
        Imgproc.cvtColor(proc, proc, Imgproc.COLOR_BGR2GRAY);
        Imgproc.blur(proc, proc, new Size(2, 2));
        Imgproc.adaptiveThreshold(proc, proc, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 121, 15);
        Core.bitwise_not(proc, proc);

        // detect line
        Mat canny = proc.clone();
        Imgproc.Canny(canny, canny, 100, 100);
        Mat lines = new Mat();
        if (VIEW_PROGRESS) {
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
        }

        // find contours
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(proc, contours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_NONE);

        // calculate center of raw img
        Point frameCenter = new Point(raw.width() / 2, raw.height() / 2);

        // extract sudoku contour
        double maxArea = 0;
        MatOfPoint sudokuContour = new MatOfPoint();
        int rawResolution = raw.rows() * raw.cols();
        for (MatOfPoint contour : contours) {
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
            double distanceFromCenter = Core.norm(new MatOfPoint(frameCenter), new MatOfPoint(contourCenter));

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

        // calculate center of sudoku contour using moment
        Moments moments = Imgproc.moments(sudokuContour, false);
        int x = (int) (moments.get_m10() / moments.get_m00());
        int y = (int) (moments.get_m01() / moments.get_m00());
        Point contourCenter = new Point(x, y);

        if (VIEW_PROGRESS) {
            Imgproc.circle(raw, contourCenter, 10, new Scalar(0, 0, 255), -1);
            Imgproc.circle(raw, frameCenter, 10, new Scalar(255, 0, 0), -1);
        }

        // calculate convex hull of contour
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
            if (VIEW_PROGRESS)
                Imgproc.drawContours(raw, hullList, 0, new Scalar(0, 255, 0), 2);
        } catch (Exception e) {
            // empty
        }

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

            if (VIEW_PROGRESS) {
                Imgproc.circle(raw, topLeft, 10, new Scalar(0, 0, 255), 2);
                Imgproc.circle(raw, topRight, 10, new Scalar(0, 0, 255), 2);
                Imgproc.circle(raw, bottomLeft, 10, new Scalar(0, 0, 255), 2);
                Imgproc.circle(raw, bottomRight, 10, new Scalar(0, 0, 255), 2);
            }

            // calculate perspective transformer
            Mat before = new MatOfPoint2f(topLeft, topRight, bottomLeft, bottomRight);
            Mat after = new MatOfPoint2f(new Point(0, 0), new Point(proc.cols(), 0),
                    new Point(0, proc.rows()), new Point(proc.cols(), proc.rows()));
            Mat perspectiveTransformer = Imgproc.getPerspectiveTransform(before, after);

            // perspective transform with processing sudoku contour
            Imgproc.warpPerspective(proc, proc, perspectiveTransformer, proc.size());

            // resize to (28 * 9) * (28 * 9) : 28 is column of train data
            Imgproc.resize(proc, proc, new Size(28 * 9, 28 * 9));
            if (VIEW_PROGRESS)
                HighGui.imshow("transform", proc);

            // split mat into 9 * 9
            Mat[] elements = new Mat[9 * 9];
            int offset = 28;
            int index = 0;
            for (int i = 0; i < 9 * offset; i += offset) {
                for (int j = 0; j < 9 * offset; j += offset)
                    elements[index++] = proc.rowRange(i, i + offset).colRange(j, j + offset);
            }

            // save with custom array
//            for (Mat cur : elements) {
//                HighGui.imshow("elements", cur);
//                HighGui.waitKey(0);
//            }

            // only used in saving train data
//            int idx = 0;
//            String savePath = "C:\\inz\\lib\\sudoku_train_data";
//            for (int i = 0; i < sudoku.length; ++i) {
//                for (int j = 0; j < sudoku[i].length; ++j) {
//                    Imgcodecs.imwrite(savePath + "\\" + sudoku[i][j] + "\\" + Math.random() + ".jpg", elements[idx++]);
//                }
//            }
//            System.out.println("save success");

            // load model
            ANN_MLP model = ANN_MLP.load("model.xml");
            Mat res = new Mat();
            Mat sudokuArrMat = new Mat();
            for (Mat cur : elements) {
                cur.convertTo(cur, CvType.CV_32FC1);
                cur = cur.reshape(cur.channels(), 1);
//                Core.normalize(cur, cur, 1, 0, Core.NORM_L2);

                // TODO : 신경망 훈련 후 스케일링 방식 교체
                for (int col = 0; col < cur.cols(); ++col)
                    cur.put(0, col, cur.get(0, col)[0] / 255.0f);
                model.predict(cur, res);

                // max col is result of model recognition
                int maxCol = -1;
                double max = -1;
                for (int col = 0; col < res.cols(); ++col) {
                    double curVal = res.get(0, col)[0];
                    if (max < curVal) {
                        max = curVal;
                        maxCol = col;
                    }
                }

                // push back to mat
                Mat curValMat = new Mat(1, 1, CvType.CV_32S);
                curValMat.put(0, 0, new int[]{maxCol});
                sudokuArrMat.push_back(curValMat);
            }

            // reshape to 1 row

            sudokuArrMat = sudokuArrMat.reshape(sudokuArrMat.channels(), 1);

            // convert mat to int arr
            int[] unsolvedSudoku = new int[sudokuArrMat.rows() * sudokuArrMat.cols()];
            for (int col = 0; col < sudokuArrMat.cols(); ++col) {
                unsolvedSudoku[col] = (int) sudokuArrMat.get(0, col)[0];
            }

            // calculate sudoku answer
            int[] sudokuAnswer = new SudokuAlgorithmSolver().getAnswer(unsolvedSudoku);
            // goto catch if answer is null

            // print sudoku answer
//            for (int cur : sudokuAnswer) {
//                System.out.print(cur + " ");
//            }
//            System.out.println();

            // get perspective transformation of raw sudoku area
            Mat perspective = new Mat();
            Imgproc.warpPerspective(raw, perspective, perspectiveTransformer, raw.size());
            int smaller = Math.min(raw.rows(), raw.cols());
            Imgproc.resize(perspective, perspective, new Size(smaller, smaller));

            // add perspective to answer text
            int rowOffset = perspective.rows() / 9;
            int colOffset = perspective.cols() / 9;
            int idx = 0;
            for (int row = 0; row < rowOffset * 9; row += rowOffset) {
                for (int col = 0; col < colOffset * 9; col += colOffset) {
//                    if(unsolvedSudoku[idx] == 0)
                        Imgproc.putText(perspective, String.valueOf(sudokuAnswer[idx]), new Point(row + 11, col + 42), Imgproc.FONT_HERSHEY_SIMPLEX, 1.3, new Scalar(0, 255, 0), 3);
                    ++idx;
                }
            }
            Imgproc.resize(perspective, perspective, raw.size());
            HighGui.imshow("perspective", perspective);

            // get inverse transform
            Mat inverse = new Mat();
            Mat inverseTransformer = Imgproc.getPerspectiveTransform(after, before);
            Imgproc.warpPerspective(perspective, inverse, inverseTransformer, raw.size());

            Imgproc.cvtColor(inverse, inverse, Imgproc.COLOR_BGR2BGRA);
            for (int row = 0; row < inverse.rows(); ++row) {
                for (int col = 0; col < perspective.cols(); ++col) {
                    if (perspective.get(row, col)[0] != 0)
                        continue;
                    if (perspective.get(row, col)[1] != 0)
                        continue;
                    if (perspective.get(row, col)[2] != 0)
                        continue;

                    inverse.put(row, col, 0, 0, 0, 0);
                }
            }

            // overlay answer number to raw perspective
            Imgproc.cvtColor(raw, raw, Imgproc.COLOR_BGR2BGRA);
            double alpha = 0.5;
            Core.addWeighted(inverse, alpha, raw, 1 - alpha, 0.0, raw);
            Imgproc.cvtColor(raw, raw, Imgproc.COLOR_BGRA2BGR);
        } catch (Exception e) {
            // empty
        }

        HighGui.imshow("cam", raw);
        HighGui.waitKey(fps);
    }
}
