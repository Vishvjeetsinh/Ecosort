#!/usr/bin/env node
// EcoSort - fetch the pretrained SSDLite MobileNetV2 (COCO) TFJS object detector.
//
// WHY this exists: the Live scan tab finds every item in a camera frame with this detector,
// then names each one with the active waste classifier. EcoSort does zero network I/O at
// runtime, so - exactly like the MobileNetV2 fallback - the weights are baked into the
// backend image at build time and refreshed on the host with `make fetch-models`.
//
// It installs to models/detectors/ssdlite_mobilenet_v2/, one level deeper than the
// classifiers, on purpose: the backend's model registry lists every *direct* child of the
// models root that holds a model.json, and a detector offered in the classifier picker
// would fail its warm-up. `detectors/` itself holds no model.json, so discovery skips it,
// while express.static still serves the files underneath it at /models/detectors/**.
//
// Usage: node scripts/fetch-detector.mjs [--dest <dir>] [--force] [--quiet]

import { runFetcher } from './lib/tfjs-model-fetch.mjs';

const BASE_URL = 'https://storage.googleapis.com/tfjs-models/savedmodel/ssdlite_mobilenet_v2/';

// Verified by downloading the upstream model: 4 x 4 MiB shards plus a 1,257,312-byte tail,
// which is also exactly what the weightsManifest arithmetic gives (239 float32 + 23 int32
// tensors).
const KNOWN_TOTAL_SHARD_BYTES = 18034528;

function buildMetadata(downloadBytes = null) {
  // docs/ARCHITECTURE.md section 5.1. frontend/src/lib/detector.js hard-codes the same
  // contract and only reads downloadBytes from here, so a missing file costs nothing but
  // the size label in the loading message.
  return {
    name: 'ssdlite_mobilenet_v2_coco',
    displayName: 'SSDLite MobileNetV2 (COCO)',
    description:
      'Finds the separate objects in a frame so each one can be classified on its own.',
    role: 'detector',
    quantization: 'none',
    downloadBytes,
    version: '1.0.0',
    createdAt: new Date().toISOString(),
    baseModel: 'SSDLite MobileNetV2',
    inputSize: 300,
    inputDtype: 'int32',
    inputRange: [0, 255],
    outputs: { scores: 'Postprocessor/Slice', boxes: 'Postprocessor/ExpandDims_1' },
    boxFormat: 'ymin-xmin-ymax-xmax, normalised to [0,1]',
    scoreActivation: 'sigmoid',
    classOffset: 1,
    labelKind: 'coco',
    classCount: 90,
    notes:
      'Pretrained COCO SSDLite MobileNetV2 graph model (the one @tensorflow-models/coco-ssd ' +
      'loads as lite_mobilenet_v2). Load with tf.loadGraphModel and run with executeAsync - ' +
      'the preprocessor is a while loop. Feed an int32 [1,H,W,3] tensor of raw 0-255 pixels ' +
      'of any size; the graph resizes it to 300x300 without keeping the aspect ratio. It ' +
      'returns [1,1917,90] per-class sigmoid scores and [1,1917,1,4] boxes, with NO ' +
      'non-max suppression applied. Score column j is COCO category id j + 1.',
  };
}

function buildSourceText(fetchedAt) {
  return [
    'EcoSort - pretrained object detector provenance',
    '===============================================',
    '',
    `Upstream URL : ${BASE_URL}`,
    'Files        : model.json, group1-shard1of5 .. group1-shard5of5',
    `Total shards : ${KNOWN_TOTAL_SHARD_BYTES} bytes`,
    'Architecture : SSDLite with a MobileNetV2 backbone, 300x300 input, COCO (90 ids, 80 used)',
    'Format       : TensorFlow.js graph model (converted TF Object Detection API SavedModel)',
    '',
    'Licence      : Apache License, Version 2.0.',
    '               The model is the ssdlite_mobilenet_v2_coco checkpoint from the',
    '               TensorFlow Object Detection API model zoo, as converted and hosted by',
    '               the tfjs-models project (@tensorflow-models/coco-ssd), both published',
    '               by Google under the Apache License 2.0:',
    '               http://www.apache.org/licenses/LICENSE-2.0',
    '               The weights are redistributed here unmodified, for offline inference.',
    '',
    `Fetched at   : ${fetchedAt}`,
    'Fetched by   : scripts/fetch-detector.mjs',
    '',
  ].join('\n');
}

await runFetcher({
  name: 'SSDLite detector',
  title: 'SSDLite MobileNetV2 (COCO) object detector',
  scriptName: 'fetch-detector.mjs',
  baseUrl: BASE_URL,
  dirName: 'detectors/ssdlite_mobilenet_v2',
  remoteFiles: [
    'model.json',
    'group1-shard1of5',
    'group1-shard2of5',
    'group1-shard3of5',
    'group1-shard4of5',
    'group1-shard5of5',
  ],
  knownTotalBytes: KNOWN_TOTAL_SHARD_BYTES,
  buildMetadata,
  buildSourceText,
  failureNote: 'The object detector was NOT installed; the Live scan tab will say so.',
});
