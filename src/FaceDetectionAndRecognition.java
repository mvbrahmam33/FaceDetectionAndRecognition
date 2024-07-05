import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_face.FaceRecognizer;
import org.bytedeco.opencv.opencv_face.LBPHFaceRecognizer;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;
import javax.swing.*;


public class FaceDetectionAndRecognition {
    public static void main(String[] args) {
        Loader.load(opencv_core.class);
        String[] names = {"brahmam", "Keanu Reeves", "ELonMusk"}; //array of names(names of persons in dataset)


        FaceRecognizer recognizer = LBPHFaceRecognizer.create();
        recognizer.read("trained_model.xml");                          //loding trained model wh using dataset


        CascadeClassifier faceCascade = new CascadeClassifier(
                "haarcascade_frontalface_default.xml");           // Load Haar Cascade for face detection
        if (faceCascade.empty()) {
            System.err.println("Failed to load Haar Cascade.");
            return;
        }


        FrameGrabber grabber = new OpenCVFrameGrabber(0);     // Open default camera
        OpenCVFrameConverter.ToMat converterToMat = new OpenCVFrameConverter.ToMat();

        try {
            grabber.start();

            CanvasFrame canvas = new CanvasFrame("Face Recognition", CanvasFrame.getDefaultGamma() / grabber.getGamma());
            canvas.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            Frame frame;
            Mat gray = new Mat();
            int[] label = new int[1];
            double[] confidence = new double[1];
            while ((frame = grabber.grab()) != null) {
                Mat mat = converterToMat.convert(frame);
                opencv_imgproc.cvtColor(mat, gray, opencv_imgproc.COLOR_BGR2GRAY);//it will turn the frame to gray
                RectVector faces = new RectVector();
                faceCascade.detectMultiScale(gray, faces);    //it is to detect the faces present in frame
                for (int i = 0; i < faces.size(); i++) {      //it is to interate over faces present in frame
                    org.bytedeco.opencv.opencv_core.Rect face = faces.get(i);
                    Mat faceROI = new Mat(gray, face);
                    recognizer.predict(faceROI, label, confidence);   //it is to recognise the faces by using pre-trained model
                    opencv_imgproc.rectangle(mat, face, new Scalar(0, 255, 0, 0));//it draws rectangle around face
                    String name;
                    if(names.length > label[0])
                        name = names[label[0]];      //it is to assign the name of person based on label
                    else
                        name = "Unknown";
                    String text = "ID: " + name ;
                    int posX = Math.max(face.tl().x() - 10, 0);//it is get location above the rectangle, so we can write text
                    int posY = Math.max(face.tl().y() - 10, 0);
                    opencv_imgproc.putText(
                            mat,
                            text,
                            new Point(posX, posY),
                            opencv_imgproc.FONT_HERSHEY_PLAIN,
                            1.0,
                            new Scalar(0, 255, 0, 0));//it is to write a name on rectangle
                }
                canvas.showImage(converterToMat.convert(mat));//shows picture in window

            }

            grabber.stop();
            canvas.dispose();

        } catch (FrameGrabber.Exception e) {
            e.printStackTrace(); // it is to catch error when something wrong happens when reading frame from camera
        }
    }
}
