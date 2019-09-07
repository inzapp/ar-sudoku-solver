package com.inzapp.arSudokuSolver.core;

import com.inzapp.arSudokuSolver.config.Config;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class AlgorithmSolver {
    /**
     *
     */
    private static class Node {
        int x;
        int y;

        Node(int x, int y) {
            this.x = x;
            this.y = y;
        }
    }

    private ExecutorService executorService;
    private boolean[][] checkCol;
    private boolean[][] checkRow;
    private boolean[][] checkBox;

    /**
     *
     */
    public AlgorithmSolver() {
        this.executorService = Executors.newSingleThreadExecutor();
    }

    /**
     * @param unsolvedSudoku
     * @param timeout
     * @return
     */
    public int[][] solveInTime(int[][] unsolvedSudoku, long timeout) {
        try {
            Callable<int[][]> callable = () -> getAnswer2d(unsolvedSudoku);
            return executorService.submit(callable).get(timeout, TimeUnit.MILLISECONDS);
        } catch (Exception e) {
            return null;
        }
    }

    /**
     * @param sudoku
     * @param cnt
     * @param nodes
     * @param idx
     * @return
     */
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

    /**
     * @param sudoku
     * @return
     */
    private int[][] getAnswer2d(int[][] sudoku) {
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