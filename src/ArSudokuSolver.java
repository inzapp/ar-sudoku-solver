import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class ArSudokuSolver {
    static {
        System.load("C:\\inzapp\\lib\\opencv\\opencv_java410.dll");
    }

    public static void main(String[] args) {
        Mat raw = Imgcodecs.imread("sudoku.jpg", Imgcodecs.IMREAD_ANYCOLOR);
        Mat proc = raw.clone();
        Imgproc.cvtColor(proc, proc, Imgproc.COLOR_BGR2GRAY);
        Imgproc.adaptiveThreshold(proc, proc, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 201, 7);
        Core.bitwise_not(proc, proc);
        Mat lines = new Mat();
        Imgproc.HoughLines(proc, lines, 1, Math.PI / 180, 350);
        for (int x = 0; x < lines.rows(); x++) {
            double rho = lines.get(x, 0)[0],
                    theta = lines.get(x, 0)[1];
            double a = Math.cos(theta), b = Math.sin(theta);
            double x0 = a * rho, y0 = b * rho;
            Point pt1 = new Point(Math.round(x0 + 1000 * (-b)), Math.round(y0 + 1000 * (a)));
            Point pt2 = new Point(Math.round(x0 - 1000 * (-b)), Math.round(y0 - 1000 * (a)));
            Imgproc.line(raw, pt1, pt2, new Scalar(0, 255, 0), 2, Imgproc.LINE_AA, 0);
        }
        System.out.println(lines.dump());

        HighGui.imshow("res", raw);
        HighGui.waitKey(0);
    }
}
