import com.sun.istack.NotNull;
import org.apache.hadoop.util.hash.Hash;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class Predictor {
    private final static Map<String, Integer> nation = new HashMap<String, Integer>() {{
        put("AUSTR", 0);
        put("BRAZ", 1);
        put("CANA", 2);
    }};

    // 评价标准
    private final static double[] precision = new double[3];
    private final static double[] recall = new double[3];
    private final static double[] f1 = new double[3];

    private final static int[] numClazz = new int[3];    // N(Ci)
    private static int numClazzSum = 0;
    private final static HashMap<String, Integer>[] numClazzTerm = new HashMap[3];  // N(Ci,Tj)
    private final static int[] numClazzNoRepeat = new int[3];    // |Vi|

    private final static int[][] confusionMatrix = new int[3][3];    // 混淆矩阵

    private final static String testDirPath = "resource/dataset/test";
    private final static String p1 = "model/job1";    // job1 的输出文件路径
    private final static String p2 = "model/job2";    // job2 的输出文件路径
    private final static String p3 = "model/job3";    // job3 的输出文件路径

    // 预测文件类别
    public static String predictFile(@NotNull String filePathStr) throws IOException {
        String predClazz = "None";   // 预测结果
        Path filePath = Paths.get(filePathStr);
        List<String> lines = Files.readAllLines(filePath);
        double[] prob = new double[3];   // 后验估计概率
        double maxProb = Double.NEGATIVE_INFINITY;   // 最大后验估计概率
        for (String clazz : nation.keySet()) {  // 遍历类型
            int nationIndex = nation.get(clazz);
            double likelihoodLogSum = 0.0;  // 似然概率取log求和
            for (String term : lines) {
                // 似然概率 P(Tj|Ci)=N(Ci,Tj)/(N(Ci)+|Vi|)
                int clazzTermNum = 0;   // clazz 类别下单词 Term 的数目
                if (numClazzTerm[nationIndex].containsKey(term)) {
                    clazzTermNum = numClazzTerm[nationIndex].get(term);
                }
                double likelihood = (clazzTermNum + 1.0) /
                        (numClazz[nationIndex] + numClazzNoRepeat[nationIndex]);
                likelihoodLogSum += Math.log(likelihood);
            }
            double experience = numClazz[nationIndex] * 1.0 / numClazzSum;    // 先验概率 P(Ci)=N(Ci)/N
            prob[nationIndex] = likelihoodLogSum + Math.log(experience);
            if (prob[nationIndex] > maxProb) {
                maxProb = prob[nationIndex];
                predClazz = clazz;
            }
        }
        return predClazz;
    }

    public static void main(String[] args) {
        for (int i = 0; i < 3; i++) {
            numClazzTerm[i] = new HashMap<>();
        }
        // 载入模型数据
        try {
            // 读取job1输出
            Path path = Paths.get(p1);
            List<String> lines = Files.readAllLines(path);
            for (String line : lines) {
                String[] kv = line.split("[ \t]+");
                int value = Integer.parseInt(kv[1]);
                numClazz[nation.get(kv[0])] = value;
                numClazzSum += value;
            }


            // 读取job2输出
            path = Paths.get(p2);
            lines = Files.readAllLines(path);
            for (String line : lines) {
                String[] kv = line.split("[ \t]+");
                String[] pair = kv[0].split("-");
                Integer num = Integer.parseInt(kv[1]);
                String clazzName = pair[0], term = pair[1];
                numClazzTerm[nation.get(clazzName)].put(term, num);
            }

            // 读取job3输出
            path = Paths.get(p3);
            lines = Files.readAllLines(path);
            for (String line : lines) {
                String[] kv = line.split("[ \t]+");
                numClazzNoRepeat[nation.get(kv[0])] = Integer.parseInt(kv[1]);
            }
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
        }

        // 计算测试集预测结果和混淆矩阵
        for (String clazz : nation.keySet()) {
            String predClazz;   // 预测的类型

            // 读取文件进行预测
            String clazzTestDirPath = testDirPath + "/" + clazz;
            File clazzTestDir = new File(clazzTestDirPath);
            File[] files = clazzTestDir.listFiles();
            if (files != null) {
                for (File file : files) {
                    // 预测类型
                    try {
                        predClazz = predictFile(file.getPath());
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                    // 更新混淆矩阵
                    ++confusionMatrix[nation.get(clazz)][nation.get(predClazz)];
                }
            }

        }

        // 评估
        double avgPrecision = 0.0, avgRecall = 0.0, avgF1 = 0.0;
        // 计算 Precision、Recall 和 F1
        for (int i = 0; i < 3; i++) {
            int TP = confusionMatrix[i][i];
            int FP = 0;
            int FN = 0;
            for (int j = 0; j < 3; j++) {
                if (j != i) {
                    FP += confusionMatrix[j][i];
                    FN += confusionMatrix[i][j];
                }
            }
            precision[i] = (double) TP / (TP + FP);
            recall[i] = (double) TP / (TP + FN);
            f1[i] = (double) 2 * precision[i] * recall[i] / (precision[i] + recall[i]);

            avgPrecision += precision[i];
            avgRecall += recall[i];
            avgF1 += f1[i];
        }
        // 计算平均值
        avgPrecision /= 3;
        avgRecall /= 3;
        avgF1 /= 3;

        System.out.format("|--------Confusion Matrix-------|\n");
        System.out.format("| Pred->| AUSTR | BRAZ  | CANA  |\n");
        System.out.format("| AUSTR |%7d|%7d|%7d|\n", confusionMatrix[0][0], confusionMatrix[0][1], confusionMatrix[0][2]);
        System.out.format("| BRAZ  |%7d|%7d|%7d|\n", confusionMatrix[1][1], confusionMatrix[1][1], confusionMatrix[1][2]);
        System.out.format("| CANA  |%7d|%7d|%7d|\n", confusionMatrix[2][0], confusionMatrix[2][1], confusionMatrix[2][2]);
        System.out.format("|-------------------------------|\n\n");

        System.out.format("|-------------Evaluation Table------------|\n");
        System.out.format("|           |  AUSTR  |  BRAZ   |  CANA   |\n");
        System.out.format("| precision |%9f|%9f|%9f|\n", precision[0], recall[0], f1[0]);
        System.out.format("| recall    |%9f|%9f|%9f|\n", precision[1], recall[1], f1[1]);
        System.out.format("| f1        |%9f|%9f|%9f|\n", precision[2], recall[2], f1[2]);
        System.out.format("|-----------------------------------------|\n");

        System.out.format("\nAverage: Precision=%f, Recall=%f, F1-Score=%f", avgPrecision, avgRecall, avgF1);
    }
}
