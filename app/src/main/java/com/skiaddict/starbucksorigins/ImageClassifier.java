package com.skiaddict.starbucksorigins;

import android.app.Activity;
import android.graphics.Bitmap;

import java.io.IOException;

/**
 * Created by jewatts on 12/9/17.
 */

class ImageClassifier {

    public static final int DIM_IMG_SIZE_X = 224;
    public static final int DIM_IMG_SIZE_Y = 224;

    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128.0f;


    public ImageClassifier(Activity activity) throws IOException {

    }

    private static int callCount = 0;

    public String classifyFrame(Bitmap bitmap) {
        try {
            callCount++;
            Thread.sleep(500);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        return "Call number: " + Integer.toString(callCount);
    }

    public void close () {

    }
}
