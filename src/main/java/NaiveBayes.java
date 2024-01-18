import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

import java.io.IOException;
import java.util.Arrays;

public class NaiveBayes {
    private final static IntWritable one = new IntWritable(1);

    public static class NoSplitTextInputFormat extends TextInputFormat {
        @Override
        protected boolean isSplitable(JobContext context, Path file) {
            return false;
        }
    }

    public static class DocClazzCountMapper extends Mapper<Object, Text, Text, IntWritable> {
        @Override
        protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            FileSplit fileSplit = (FileSplit) context.getInputSplit();
            Text clazzName = new Text(fileSplit.getPath().getParent().getName());
            context.write(clazzName, one);
        }
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

    public static class TermClazzCountMapper extends Mapper<Object, Text, Text, IntWritable> {
        @Override
        protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
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

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        if ((otherArgs.length <= 3)) {
            System.err.println("Usage: NaiveBayes <in> <job1-out> <job2-out> <class1, [class2, ...]>");
            System.exit(2);
        }

        Job job1 = Job.getInstance(conf, "统计不同类别的文章数");
        job1.setJarByClass(NaiveBayes.class);
        job1.setInputFormatClass(NoSplitTextInputFormat.class);
        job1.setMapperClass(DocClazzCountMapper.class);
        job1.setReducerClass(DocClazzCountReducer.class);
        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(IntWritable.class);

        Job job2 = Job.getInstance(conf, "统计特定类别下不同单词的词频");
        job2.setJarByClass(NaiveBayes.class);
        job2.setInputFormatClass(NoSplitTextInputFormat.class);
        job2.setMapperClass(TermClazzCountMapper.class);
        job2.setReducerClass(TermClazzCountReducer.class);
        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(IntWritable.class);

        String[] clazzName = Arrays.copyOfRange(otherArgs, 3, otherArgs.length);
        for (String s : clazzName) {
            Path inputPath = new Path(otherArgs[0] + "/" + s);
            FileInputFormat.addInputPath(job1, inputPath);
            FileInputFormat.addInputPath(job2, inputPath);
        }


        Path outputPath1 = new Path(otherArgs[1]);
        FileOutputFormat.setOutputPath(job1, outputPath1);
        Path outputPath2 = new Path(otherArgs[2]);
        FileOutputFormat.setOutputPath(job2, outputPath2);

        System.exit(job1.waitForCompletion(true) && job2.waitForCompletion(true) ? 0 : 1);
    }
}
