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

    private final static int[] numClazz = new int[3];    // N(Ci)
    private final static HashMap<String, Integer>[] numClazzTerm = new HashMap[3];  // N(Ci,Tj)
    private final static int[] numClazzNoRepeat = new int[3];    // |Vi|

    private final static int[][] confusionMatrix = new int[3][3];    // 混淆矩阵

    private final static String p1 = "model/job1";    // job1 的输出文件路径
    private final static String p2 = "model/job2";    // job2 的输出文件路径
    private final static String p3 = "model/job3";    // job3 的输出文件路径

    public static void main(String[] args) {
        // 载入模型数据
        try {
            // 读取job1输出
            Path path = Paths.get(p1);
            List<String> lines = Files.readAllLines(path);
            for (String line : lines) {
                String[] kv = line.split("[ \t]+");
                numClazz[nation.get(kv[0])] = Integer.parseInt(kv[1]);
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
        }

        // 计算测试集预测结果和混淆矩阵 TODO

        // 评估 TODO
    }
}
