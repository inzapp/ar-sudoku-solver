import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.*;

class Main {
    private static boolean[][] check_col = new boolean[9][10];//행 검사
    private static boolean[][] check_row = new boolean[9][10];//열 검사
    private static boolean[][] check_box = new boolean[9][10];//박스 검사

    static class Node {
        int x;
        int y;

        Node(int x, int y) {
            this.x = x;
            this.y = y;
        }
    }

    // arr=스도쿠, count= 전체 빈칸의 수, nodes=빈칸위치를 담고있는 리스트, idx=현재 탐색 중인 빈칸의 인덱스
    private static boolean solve(int[][] arr, int count, ArrayList<Node> nodes, int idx) {
        if (idx >= count) {//비어있는 칸을 모두 탐색했다면 출력
            for (int i = 0; i < 9; i++) {
                for (int j = 0; j < 9; j++) {
                    System.out.print(arr[i][j] + " ");
                }
                System.out.println();
            }
            return true;//탐색 끝
        }
        Node node = nodes.get(idx);
        for (int i = 1; i <= 9; i++) {//1부터 9까지 모든 경우의 수 탐색
//promising?
            if (check_col[node.x][i])
                continue;
            if (check_row[node.y][i])
                continue;
            if (check_box[(node.x / 3) * 3 + (node.y) / 3][i])
                continue;
            check_col[node.x][i] = true;
            check_row[node.y][i] = true;
            check_box[(node.x / 3) * 3 + (node.y) / 3][i] = true;

            arr[node.x][node.y] = i;

            if (solve(arr, count, nodes, idx + 1))
                return true;//탐색 끝
//백트래킹
            arr[node.x][node.y] = 0;
            check_col[node.x][i] = false;
            check_row[node.y][i] = false;
            check_box[(node.x / 3) * 3 + (node.y) / 3][i] = false;
        }

        return false;
    }

    static void main_(String[] args) {
        Scanner scan = new Scanner(System.in);
        int[][] sudoku = new int[9][9];

        int count = 0;
        ArrayList<Node> nodes = new ArrayList<>();

        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                sudoku[i][j] = scan.nextInt();
                if (sudoku[i][j] == 0) {
                    count++;
                    nodes.add(new Node(i, j));
                } else {
                    check_col[i][sudoku[i][j]] = true;
                    check_row[j][sudoku[i][j]] = true;
                    check_box[(i / 3) * 3 + (j / 3)][sudoku[i][j]] = true;
                }
            }
        }

        solve(sudoku, count, nodes, 0);
    }
}

class SudokuAlgorithmSolver {
    private static int cnt = 0;
    private static List<Integer> existValue = new ArrayList<>();

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

        boolean removeCandidate(int val) {
            boolean res = candidate.remove(new Integer(val));
            if (candidate.size() == 1)
                this.val = candidate.get(0);
            return res;
        }
    }

    void solve() {
        // input value
        Scanner sc = new Scanner(System.in);
        Cell[][] pan = new Cell[9][9];
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j)
                pan[i][j] = new Cell(sc.nextInt());
        }

        // brute force
        bruteForce(pan, 0, 0);

        // print res
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j)
                System.out.print(pan[i][j].val + " ");
            System.out.println();
        }
    }

    private void bruteForce(Cell[][] pan, int row, int col) {
        // remove already exist value from candidate
        while (true) {
            boolean isCandidateChanged = false;
            for (int i = 0; i < 9; ++i) {
                for (int j = 0; j < 9; ++j) {
                    getExistValue(pan, i, j);
                    for (int cur : existValue) {
                        if (pan[i][j].removeCandidate(cur))
                            isCandidateChanged = true;
                    }
                }
            }
            if (!isCandidateChanged)
                break;
        }

        // count non zero, return if non zero count is 81 (all solved)
        int nonZeroCnt = 0;
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                if (pan[i][j].val != 0)
                    ++nonZeroCnt;
                else break;
            }
        }

        // validation check
        if (nonZeroCnt == 81) {
            boolean condition = true;

            // row validation check
            int sum;
            for (int i = 0; i < 9; ++i) {
                sum = 0;
                for (int cur : getHorizontal(pan, i))
                    sum += cur;
                System.out.println(sum);
                if (sum != 45) {
                    condition = false;
                    break;
                }
            }

            // col validation check
            for (int i = 0; i < 9; ++i) {
                sum = 0;
                for (int cur : getVertical(pan, i))
                    sum += cur;
                if (sum != 45) {
                    condition = false;
                    break;
                }
            }

            // square validation check
            int[] index = new int[]{1, 4, 7};
            for (int i : index) {
                for (int j : index) {
                    sum = 0;
                    for (int cur : getSquare(pan, i, j))
                        sum += cur;
                    if (sum != 45) {
                        condition = false;
                        break;
                    }
                }
            }

            System.out.println(condition);
            if (condition)
                return;
        }

        for (int i = row; i < 9; ++i) {
            for (int j = col; j < 9; ++j) {
                if (pan[i][j].candidate.size() == 0)
                    continue;

                if (pan[i][j].candidate.size() == 2) {
                    for (int k = 0; k < pan[i][j].candidate.size(); ++k) {
                        pan[i][j].removeCandidate(k);
                        bruteForce(pan, i, j);
                        pan[i][j].candidate.add(k);
                    }
                }
            }
        }


    }

    private void getExistValue(Cell[][] pan, int row, int col) {
        if (existValue.size() > 0) {
            existValue.subList(0, existValue.size()).clear();
        }
        existValue.addAll(getHorizontal(pan, row));
        existValue.addAll(getVertical(pan, col));
        existValue.addAll(getSquare(pan, row, col));
    }

    private List<Integer> getHorizontal(Cell[][] pan, int row) {
        List<Integer> horizontal = new ArrayList<>();
        for (int i = 0; i < 9; ++i)
            horizontal.add(pan[row][i].val);
        return horizontal;
    }

    private List<Integer> getVertical(Cell[][] pan, int col) {
        List<Integer> vertical = new ArrayList<>();
        for (int i = 0; i < 9; ++i)
            vertical.add(pan[i][col].val);
        return vertical;
    }

    private List<Integer> getSquare(Cell[][] pan, int row, int col) {
        if (row == 0 || row == 1 || row == 2)
            row = 0;
        else if (row == 3 || row == 4 || row == 5)
            row = 3;
        else if (row == 6 || row == 7 || row == 8)
            row = 6;

        if (col == 0 || col == 1 || col == 2)
            col = 0;
        else if (col == 3 || col == 4 || col == 5)
            col = 3;
        else if (col == 6 || col == 7 || col == 8)
            col = 6;

        int iEnd = row + 3;
        int jEnd = col + 3;
        List<Integer> square = new ArrayList<>();
        for (int i = row; i < iEnd; ++i) {
            for (int j = col; j < jEnd; ++j)
                square.add(pan[i][j].val);
        }
        return square;
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
        Main.main_(new String[]{});
//        new SudokuAlgorithmSolver().solve();
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
