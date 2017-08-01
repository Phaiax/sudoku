package de.invisibletower.sudoku;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.imgproc.Imgproc;

import android.os.Bundle;
import android.provider.ContactsContract;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.WindowManager;

import static org.opencv.core.CvType.CV_8UC1;

public class PrototypeActivity extends AppCompatActivity implements CvCameraViewListener2 {
    private static final String TAG = "OCVSample::Activity";

    private static final int       VIEW_MODE_RGBA     = 0;
    private static final int       VIEW_MODE_GRAY     = 1;
    private static final int       VIEW_MODE_CANNY    = 2;
    private static final int       VIEW_MODE_DEBUG1    = 4;
    private static final int       VIEW_MODE_FEATURES = 5;

    private int                    mViewMode;
    private Mat                    mRgba;
    private Mat                    mRgbaSmall;
    private Mat                    mIntermediateMat;
    private Mat                    mGray;

    private CameraBridgeViewBase mOpenCvCameraView;
    private Recognizer recognizer;


    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public PrototypeActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.tutorial1_surface_view);

        Toolbar myToolbar = (Toolbar) findViewById(R.id.my_toolbar);
        setSupportActionBar(myToolbar);

        recognizer = new Recognizer();
        mViewMode = VIEW_MODE_DEBUG1;

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.tutorial1_activity_java_surface_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);



//        Button next_step = (Button) findViewById(R.id.next_example_step);
//        next_step.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View v) {
//
//            }
//        });
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mIntermediateMat = new Mat(height, width, CvType.CV_8UC4);
        mGray = new Mat(height, width, CvType.CV_8UC1);
    }

    public void onCameraViewStopped() {
        mRgba.release();
        mGray.release();
        mIntermediateMat.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        mGray = inputFrame.gray();
        recognizer.recognize(mGray);

        final int viewMode = mViewMode;
        switch (viewMode) {
            case VIEW_MODE_GRAY:
                // input frame has gray scale format
                Imgproc.cvtColor(inputFrame.gray(), mRgba, Imgproc.COLOR_GRAY2RGBA, 4);
                break;
            case VIEW_MODE_RGBA:
                // input frame has RBGA format
                mRgba = inputFrame.rgba();
                break;
            case VIEW_MODE_CANNY:
                // input frame has gray scale format
                mRgba = inputFrame.rgba();
                Imgproc.Canny(inputFrame.gray(), mIntermediateMat, 80, 100);
                Imgproc.cvtColor(mIntermediateMat, mRgba, Imgproc.COLOR_GRAY2RGBA, 4);
                break;
            case VIEW_MODE_DEBUG1:
                Imgproc.resize(recognizer.getDebugImage(), mGray, mGray.size());
                Imgproc.cvtColor(mGray, mRgba, Imgproc.COLOR_GRAY2RGBA, 4);
                break;
            case VIEW_MODE_FEATURES:
                // input frame has RGBA format
                mRgba = inputFrame.rgba();
                mGray = inputFrame.gray();
                // FindFeatures(mGray.getNativeObjAddr(), mRgba.getNativeObjAddr());
                break;
        }

        return mRgba;
    }


    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        getMenuInflater().inflate(R.menu.default_menu, menu);
        return super.onCreateOptionsMenu(menu);
    }

    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
        switch (item.getItemId()) {
            case R.id.action_gray:
                mViewMode = VIEW_MODE_GRAY;
                break;
            case R.id.action_rgb:
                mViewMode = VIEW_MODE_RGBA;
                break;
            case R.id.action_canny:
                mViewMode = VIEW_MODE_CANNY;
                break;
            case R.id.action_debug1:
                mViewMode = VIEW_MODE_DEBUG1;
                break;
        }
//        } else if (item == mItemPreviewCanny) {
//            mViewMode = VIEW_MODE_CANNY;
//        } else if (item == mItemPreviewFeatures) {
//            mViewMode = VIEW_MODE_FEATURES;
//        }

        return true;
    }
}
