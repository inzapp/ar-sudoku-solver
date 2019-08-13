import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class ArSudokuSolver {
    static {
        System.load("C:\\inzapp\\lib\\opencv\\opencv_java410.dll");
    }

    public static void main(String[] args) {
        // load image
        Mat raw = Imgcodecs.imread("sudoku.jpg", Imgcodecs.IMREAD_ANYCOLOR);
        Mat proc = raw.clone();

        // pre processing
        Imgproc.cvtColor(proc, proc, Imgproc.COLOR_BGR2GRAY);
        Imgproc.adaptiveThreshold(proc, proc, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 201, 7);
        Core.bitwise_not(proc, proc);

        // detect line
        Mat lines = new Mat();
        Imgproc.HoughLines(proc, lines, 1, Math.PI / 180, 350);
        for (int x = 0; x < lines.rows(); x++) {
            double rho = lines.get(x, 0)[0];
            double theta = lines.get(x, 0)[1];
            double a = Math.cos(theta);
            double b = Math.sin(theta);
            double x0 = a * rho;
            double y0 = b * rho;
            Point pt1 = new Point(Math.round(x0 + 1000 * (-b)), Math.round(y0 + 1000 * (a)));
            Point pt2 = new Point(Math.round(x0 - 1000 * (-b)), Math.round(y0 - 1000 * (a)));
//            Imgproc.line(raw, pt1, pt2, new Scalar(0, 255, 0), 1, Imgproc.LINE_AA, 0);
        }

        // detect sudoku rectangle
        // 1. find contours
        List<MatOfPoint> contourList = new ArrayList<>();
        Imgproc.findContours(proc, contourList, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        for(int i=0; i<contourList.size(); ++i){
//            Imgproc.drawContours(raw, contourList, i, new Scalar(0, 255, 0), 2);
        }

        // 2. convert contours to rect
        List<Rect> rectList = new ArrayList<>();
        MatOfPoint2f curPoly = new MatOfPoint2f();
        for (MatOfPoint contour : contourList) {
            Imgproc.approxPolyDP(new MatOfPoint2f(contour.toArray()), curPoly, 1, true);
            rectList.add(Imgproc.boundingRect(curPoly));
        }

        for(Rect cur : rectList) {
            Imgproc.rectangle(raw, cur, new Scalar(0, 255, 0), 2);
        }

        HighGui.imshow("res", raw);
        HighGui.waitKey(0);
    }
}
