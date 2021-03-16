const sharp = require('sharp');
const tfjs = require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs-core");
const posenet = require("@tensorflow-models/posenet");
const cv = require('./opencv.js');
const kp_score = 0.5;
const p_score = 0.2;
const cnn_size = 128;
const pairs = [
    [1, 3], [2, 4],
    [1, 0], [2, 0],
    [5, 6], [5, 7],
    [7, 9], [6, 8],
    [8, 10], [5, 11],
    [6, 12], [11, 12],
    [11, 13], [13, 15],
    [12, 14], [14, 16]
]
const labels = ['hunched', 'left', 'right', 'good'];

/**
 * Convert image to tensor, which can be fed into Tensorflow model.
 * @param {*} imgPath path to the image.
 * @returns tensor.
 */
async function getTensorFromImgPath(imgPath) {
    console.log(imgPath);
    let data = await sharp(imgPath).rotate().toBuffer();
    let tensor = tfjs.node.decodeImage(data); // equals to tfjs.decodeJPEG() of React Native Backend.
    return tensor;
}

/**
 * Get keypoints of person pose.
 * @param {*} net Posenet.
 * @param {*} tensor input tensor.
 * @returns keypoints.
 */
async function getKeypoints(net, tensor) {
    let value = await net.estimateSinglePose(tensor, {
        flipHorizontal: false
    });
    if (value.score > p_score) {
        let selected = value.keypoints.filter(a => {
            return a.score > kp_score;
        });
        let x_ = selected.map(a => a.position.x);
        let y_ = selected.map(a => a.position.y);
        let max_x = Math.max.apply(Math, x_);
        let min_x = Math.min.apply(Math, x_);
        let max_y = Math.max.apply(Math, y_);
        let min_y = Math.min.apply(Math, y_);
        value.keypoints = value.keypoints.map(a => {
            a.position.x = (a.position.x - min_x) / (max_x - min_x) * cnn_size;
            a.position.y = (a.position.y - min_y) / (max_y - min_y) * cnn_size;
            a.position.x = Math.floor(a.position.x);
            a.position.y = Math.floor(a.position.y);
            return a;
        });
        return value;
    }
}

/**
 * Create tensor from keypoints to feed into classifying model.
 * @param {*} value keypoints
 * @returns tensor.
 */
async function makeTensorFromKeypoints(value) {
    let mat = cv.Mat.zeros(cnn_size, cnn_size, 16);
    pairs.forEach(pair => {
        if (value.keypoints[pair[0]].score > kp_score &&
            value.keypoints[pair[1]].score > kp_score) {
            let bgn = new cv.Point(value.keypoints[pair[0]].position.x, value.keypoints[pair[0]].position.y);
            let end = new cv.Point(value.keypoints[pair[1]].position.x, value.keypoints[pair[1]].position.y);
            cv.line(mat, bgn, end, new cv.Scalar(0, 255, 0), 2);
            cv.ellipse(mat, bgn, new cv.Size(2, 2), 0, 0, 360, new cv.Scalar(0, 0, 255), cv.FILLED);
            cv.ellipse(mat, end, new cv.Size(2, 2), 0, 0, 360, new cv.Scalar(0, 0, 255), cv.FILLED);
        }
    });
    let tensor = tf.tensor(mat.data, [1, mat.rows, mat.cols, 3], "int32");
    return tensor;
}
/**
 * Predict the posture of the person in the image.
 * @param {*} img_path path to the image.
 */
async function predict(img_path) {
    const pose = await posenet.load({
        architecture: 'MobileNetV1',
        outputStride: 16,
        inputResolution: 257,
        multiplier: 0.5
    });
    const classify = await tfjs.loadLayersModel('file://../models/tfjs/model.json');

    // Process: image -> tensor -> keypoints -> tensor -> result.
    let inputTensor = await getTensorFromImgPath(img_path);
    let keypoints = await getKeypoints(pose, inputTensor);
    let poseTensor = await makeTensorFromKeypoints(keypoints);
    let result = classify.predict(poseTensor);
    result = tf.argMax(result, 1);
    let tensorData = result.dataSync();
    console.log("RESULT:", labels[tensorData[0]]);
}

predict('/media/trunghieu/H.Trung/Tung/pose_demo/data/posedata/1/hunched/20210226_164820.jpg');