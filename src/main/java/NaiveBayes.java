import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

import java.io.IOException;
import java.util.Arrays;

public class NaiveBayes {
    private final static IntWritable one = new IntWritable(1);

    public static class parentDirNameRecordReader extends RecordReader<Text, IntWritable> {
        private FileSplit fileSplit;
        private boolean processed = false;  // 是否处理了该文件
        private Text parentDirName;

        @Override
        public void initialize(InputSplit inputSplit, TaskAttemptContext taskAttemptContext) throws IOException, InterruptedException {
            fileSplit = (FileSplit) inputSplit;
        }

        @Override
        public boolean nextKeyValue() throws IOException, InterruptedException {
            // 如果没有处理文件，则提取split的文件名，并标记为已经处理
            if (!processed) {
                parentDirName = new Text(fileSplit.getPath().getParent().getName());
                processed = true;
                return true;
            }
            return false;
        }

        @Override
        public Text getCurrentKey() throws IOException, InterruptedException {
            return parentDirName;
        }

        @Override
        public IntWritable getCurrentValue() throws IOException, InterruptedException {
            return new IntWritable(1);
        }

        @Override
        public float getProgress() throws IOException, InterruptedException {
            // 由于一次处理一个文件，因此进度只有0%和100%两种状态，分别为未处理和已处理
            return processed ? 1.0f : 0.0f;
        }

        @Override
        public void close() throws IOException {
        }
    }

    // 一次读入一个文件，返回文件所在文件夹名和1的键值对
    public static class NoSplitParentDirNameInputFormat extends FileInputFormat<Text, IntWritable> {

        // 不要切分文件，否则会统计错误
        @Override
        protected boolean isSplitable(JobContext context, Path filename) {
            return false;
        }

        @Override
        public RecordReader<Text, IntWritable> createRecordReader(InputSplit inputSplit, TaskAttemptContext taskAttemptContext) throws InterruptedException, IOException{
            parentDirNameRecordReader reader = new parentDirNameRecordReader();
            reader.initialize(inputSplit, taskAttemptContext);
            return reader;
        }
    }

    public static class DocClazzCountMapper extends Mapper<Text, IntWritable, Text, IntWritable> {
        // 留空，继承NoSplitParentDirNameInputFormat输出的KV
    }

    public static class DocClazzCountReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        @Override
        protected void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int amount = 0;
            for (Writable v : values) {
                ++amount;
            }
            context.write(key, new IntWritable(amount));
        }
    }

    public static class TermClazzCountMapper extends Mapper<LongWritable, Text, Text, IntWritable> {
        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            FileSplit fileSplit = (FileSplit) context.getInputSplit();
            Text clazzName = new Text(fileSplit.getPath().getParent().getName());
            while (context.nextKeyValue()) {
                Text term = context.getCurrentValue();
                Text pair = new Text(clazzName.toString() + "-" + term.toString());
                context.write(pair, one);
            }
        }
    }

    public static class TermClazzCountReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        @Override
        protected void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int amount = 0;
            for (IntWritable v : values) {
                ++amount;
            }
            context.write(key, new IntWritable(amount));
        }
    }

    // 统计每个类别下的不重复单词数
    public static class NoRepeatTermCountMapper extends Mapper<LongWritable, Text, Text, IntWritable> {
        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String str = value.toString();
            String[] kv = str.split("[ \t]+");  // 分离键值对
            String[] pair = kv[0].split("-");   // 从类别和单词名的分隔符位置提取类别名和单词名
            context.write(new Text(pair[0]), new IntWritable(Integer.parseInt(kv[1])));
        }
    }

    public static class NoRepeatTermCountReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        @Override
        protected void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int amt = 0;
            for (IntWritable v : values) {
                amt += v.get();
            }
            context.write(key, new IntWritable(amt));
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        if ((otherArgs.length <= 4)) {
            System.err.println("Usage: NaiveBayes <in> <job1-out> <job2-out> <job3-out> <class1, [class2, ...]>");
            System.exit(2);
        }

        Job job1 = Job.getInstance(conf, "统计不同类别的文章数");
        job1.setJarByClass(NaiveBayes.class);
        job1.setInputFormatClass(NoSplitParentDirNameInputFormat.class);
        job1.setMapperClass(DocClazzCountMapper.class);
        job1.setReducerClass(DocClazzCountReducer.class);
        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(IntWritable.class);

        Job job2 = Job.getInstance(conf, "统计特定类别下不同单词的词频");
        job2.setJarByClass(NaiveBayes.class);
        job2.setInputFormatClass(TextInputFormat.class);
        job2.setMapperClass(TermClazzCountMapper.class);
        job2.setReducerClass(TermClazzCountReducer.class);
        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(IntWritable.class);

        Job job3 = Job.getInstance(conf, "统计不同类别文章中不重复单词集合的大小");
        job3.setJarByClass(NaiveBayes.class);
        job3.setInputFormatClass(TextInputFormat.class);
        job3.setMapperClass(NoRepeatTermCountMapper.class);
        job3.setReducerClass(NoRepeatTermCountReducer.class);
        job3.setOutputKeyClass(Text.class);
        job3.setOutputValueClass(IntWritable.class);

        String[] clazzName = Arrays.copyOfRange(otherArgs, 4, otherArgs.length);
        for (String s : clazzName) {
            Path inputPath = new Path(otherArgs[0] + "/" + s);
            FileInputFormat.addInputPath(job1, inputPath);
            FileInputFormat.addInputPath(job2, inputPath);
        }

        Path outputPath1 = new Path(otherArgs[1]);
        FileOutputFormat.setOutputPath(job1, outputPath1);
        Path outputPath2 = new Path(otherArgs[2]);
        FileOutputFormat.setOutputPath(job2, outputPath2);
        Path outputPath3 = new Path(otherArgs[3]);
        FileOutputFormat.setOutputPath(job3, outputPath3);

        int flag = job1.waitForCompletion(true) && job2.waitForCompletion(true) ? 0 : 1;
        if (flag != 0) System.exit(flag);

        FileInputFormat.addInputPath(job3, outputPath2);    // 将作业2的输出作为作业3的输入
        System.exit(job3.waitForCompletion(true) ? 0 : 1);
    }
}
