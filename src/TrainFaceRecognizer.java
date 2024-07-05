import org.bytedeco.javacpp.Loader;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.opencv_face.FaceRecognizer;
import org.bytedeco.opencv.opencv_face.LBPHFaceRecognizer;
import java.io.File;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.List;

public class TrainFaceRecognizer {
    public static void main(String[] args) {
        Loader.load(opencv_core.class);

        String trainingDir = "dataset";//it is the path of dataset
        File root = new File(trainingDir);


        File[] subDirs = root.listFiles(File::isDirectory); // List subdirectories for each person
        List<Mat> imagesList = new ArrayList<>();
        List<Integer> labelsList = new ArrayList<>();


        for (File subDir : subDirs) {  // Iterate through each person directory
            File[] imageFiles = subDir.listFiles((dir, name) -> name.toLowerCase().endsWith(".jpg"));
            int label;
            try {
                label = Integer.parseInt(subDir.getName()); // Assuming directory name is the label
            } catch (NumberFormatException e) {
                System.err.println("Directory name must be an integer representing the label: " + subDir.getName());
                continue;
            }

            for (File image : imageFiles) {
                Mat img = opencv_imgcodecs.imread(image.getAbsolutePath(), opencv_imgcodecs.IMREAD_GRAYSCALE);
                imagesList.add(img);
                labelsList.add(label);
            }
        }


        MatVector images = new MatVector(imagesList.size());  // Convert list to MatVector and labels Mat
        Mat labels = new Mat(imagesList.size(), 1, opencv_core.CV_32SC1);
        IntBuffer labelsBuf = labels.createBuffer();

        for (int i = 0; i < imagesList.size(); i++) {
            images.put(i, imagesList.get(i));
            labelsBuf.put(i, labelsList.get(i));
        }


        FaceRecognizer faceRecognizer = LBPHFaceRecognizer.create();   //it is to train the face recognizer
        faceRecognizer.train(images, labels);


        faceRecognizer.save("trained_model.xml");  // it is to Save the trained model

        System.out.println("Model training completed and it is saved as trained_model.xml");
    }
}
