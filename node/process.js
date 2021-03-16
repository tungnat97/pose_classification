"use strict";
const path = require("path");
const Jimp = require('jimp');
const sharp = require('sharp');
const fs = require('fs');

const { writeFile, existsSync, mkdirSync, readdir, readdirSync, readFileSync } = require("fs")
const tfjs = require("@tensorflow/tfjs-node-gpu");
const tf = require("@tensorflow/tfjs-core");
const { canvas, Image, createCanvas } = require("canvas");
const posenet = require("@tensorflow-models/posenet");
const cv = require('./opencv.js');

const DATA_DIR = path.join(__dirname, "..", "data", "posedata");
const PROCESS_DIR = path.join(__dirname, "..", "data", "cnn", "posenet");
const FAKE_DIR = path.join(__dirname, "..", "data", "cnn", "fake");
const kp_score = 0.5;
const p_score = 0.2;
const cnn_size = 128;
const synth = 5;

const parts = {
    "nose": 0,
    "leftEye": 1,
    "rightEye": 2,
    "leftEar": 3,
    "rightEar": 4,
    "leftShoulder": 5,
    "rightShoulder": 6,
    "leftElbow": 7,
    "rightElbow": 8,
    "leftWrist": 9,
    "rightWrist": 10,
    "leftHip": 11,
    "rightHip": 12,
    "leftKnee": 13,
    "rightKnee": 14,
    "leftAnkle": 15,
    "rightAnkle": 16
}

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

function getRndInteger(min, max) {
    return Math.floor(Math.random() * (max - min)) + min;
}

function createDataDir(prd) {
    if (!existsSync(prd)) {
        mkdirSync(prd);
    }
    readdir(DATA_DIR, (err, people) => {
        if (err) {
            console.error("Error read", DATA_DIR);
        } else {
            people.forEach(person => {
                if (!existsSync(path.join(prd, person))) {
                    mkdirSync(path.join(prd, person));
                }
                readdir(path.join(DATA_DIR, person), (err, poses) => {
                    if (err) {
                        console.error("Error read person", person);
                    } else {
                        console.log("Reading person pose", person);
                        poses.forEach(pose => {
                            if (!existsSync(path.join(prd, person, pose))) {
                                mkdirSync(path.join(prd, person, pose));
                            }
                        });
                    }
                });
            });
        }
    });
}

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
        return value
    }
}

async function drawSkeletonFromKeypoints(value, savepath) {
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
    let x = await Jimp.read({
        width: mat.cols,
        height: mat.rows,
        data: Buffer.from(mat.data)
    });
    x.write(savepath);
    console.log(savepath);
    mat.delete();
}

function saveJSON(value, savepath) {
    let jsonData = JSON.stringify(value);
    fs.writeFile(savepath, jsonData, function (err) {
        if (err) {
            console.log(err);
        }
    });
}

function generateSyntheticData(value) {
    value.keypoints.map(a => {
        a.position.x = a.position.x + getRndInteger(-synth, synth);
        a.position.y = a.position.y + getRndInteger(-synth, synth);
    });
    return value;
}


async function process(fake, num_fake = 0) {
    const net = await posenet.load({
        architecture: 'MobileNetV1',
        outputStride: 16,
        inputResolution: 257,
        multiplier: 0.5
    });

    readdir(DATA_DIR, (err, people) => {
        people.forEach(person => {
            readdir(path.join(DATA_DIR, person), (err, poses) => {
                poses.forEach(pose => {
                    readdir(path.join(DATA_DIR, person, pose), (err, imgs) => {
                        let cnt = 0;
                        imgs.forEach(function (img, idx) {
                            const filepath = path.join(DATA_DIR, person, pose, img);
                            const savepath = path.join(PROCESS_DIR, person, pose, `${idx}.jpg`);
                            sharp(filepath).rotate().toBuffer().then(async buffer => {
                                let tensor = tfjs.node.decodeImage(buffer);
                                let v = await getKeypoints(net, tensor);
                                if (fake == false) {
                                    drawSkeletonFromKeypoints(v, savepath);
                                } else {
                                    for (let i = 0; i < num_fake; ++i) {
                                        let fake_ = path.join(FAKE_DIR, person, pose, `${cnt}.jpg`);
                                        let cv = JSON.parse(JSON.stringify(v));
                                        let fake_value = generateSyntheticData(cv);
                                        drawSkeletonFromKeypoints(fake_value, fake_);
                                        ++cnt;
                                    }
                                }
                                tfjs.dispose(tensor);
                            });
                        });
                    });
                });
            });
        });
    });
}

createDataDir(PROCESS_DIR);
createDataDir(FAKE_DIR);
process(true, 100).catch(err => {
    console.error(err)
});
