package com.inzapp.arSudokuSolver.util;

import com.inzapp.arSudokuSolver.config.Config;
import com.inzapp.arSudokuSolver.config.ViewMode;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;

public class View {
    public static Mat pureResult;
    public static Mat progress;
    public static Mat sudokuArea;
    public static Mat transform;
    public static Mat[][] elements = new Mat[9][9];

    /**
     * initialize view reference to zero mat
     */
    public static void init() {
        if (Config.VIEW_PROGRESS) {
            pureResult = Mat.zeros(new Size(854, 480), CvType.CV_8UC3);
            progress = Mat.zeros(new Size(854, 480), CvType.CV_8UC3);
            sudokuArea = Mat.zeros(new Size(854, 480), CvType.CV_8UC1);
            transform = Mat.zeros(new Size(28 * 15, 28 * 15), CvType.CV_8UC1);
            for (int i = 0; i < elements.length; ++i) {
                for (int j = 0; j < elements[i].length; ++j)
                    elements[i][j] = Mat.zeros(new Size(28 * 15 / 9, 28 * 15 / 9), CvType.CV_8UC1);
            }
        }
    }

    /**
     * shows the values stored in the static reference at once
     *
     * @param fps specified frame per second
     */
    public static void show(int fps) {
        imshow("cam", pureResult, 0, 0);
        if (Config.VIEW_PROGRESS) {
            if (Config.VIEW_MODE == ViewMode.VIEW_MODE_PROGRESS) {
                imshow("progress", progress, 870, 0);
                imshow("sudoku_area", sudokuArea, 0, 525);
                imshow("transform", transform, 870, 525);
            } else if (Config.VIEW_MODE == ViewMode.VIEW_MODE_ELEMENTS) {
                imshow("transform", transform, 435, 525);
                int offset = 110;
                for (int i = 0; i < elements.length; ++i) {
                    for (int j = 0; j < elements[i].length; ++j)
                        imshow("e" + i + j, elements[i][j], 880 - offset + (j * offset + offset), (i * offset + offset) - offset);
                }
            }
        }
        HighGui.waitKey(fps);
    }

    /**
     * custom imshow function to use instead of HighGui.imshow()
     *
     * @param windowName window name of image
     * @param img        image of mat type
     * @param x          x coordinate of image window
     * @param y          y coordinate of image window
     */
    private static void imshow(String windowName, Mat img, int x, int y) {
        HighGui.namedWindow(windowName, HighGui.WINDOW_AUTOSIZE);
        HighGui.moveWindow(windowName, x, y);
        HighGui.imshow(windowName, img);
    }
}
