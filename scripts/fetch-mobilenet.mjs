#!/usr/bin/env node
// EcoSort - fetch the pretrained MobileNetV2 (1.0, 224) TFJS graph model.
//
// WHY this exists: EcoSort does zero network I/O at runtime, so the fallback engine's
// weights must already be on disk when the app boots. The backend image bakes them in at
// build time; humans refresh them on the host with `make fetch-models`. The download,
// verification and atomic install live in scripts/lib/tfjs-model-fetch.mjs, shared with
// fetch-detector.mjs; this file only describes the model. Neither has npm dependencies, so
// both run inside a bare `node:22-bookworm-slim` build layer before `npm install`.
//
// Usage: node scripts/fetch-mobilenet.mjs [--dest <dir>] [--force] [--quiet]

import { runFetcher } from './lib/tfjs-model-fetch.mjs';

const BASE_URL =
  'https://storage.googleapis.com/tfjs-models/savedmodel/mobilenet_v2_1.0_224/';

// Verified by downloading the upstream model (docs/ARCHITECTURE.md section 2.1). Used as a
// second opinion when the weightsManifest arithmetic cannot be trusted.
const KNOWN_TOTAL_SHARD_BYTES = 13984940;

function buildMetadata(downloadBytes = null) {
  // Shape is fixed by docs/ARCHITECTURE.md section 2.3 so classifier.js keeps exactly one
  // code path across both engines.
  return {
    name: 'mobilenet_v2_1.0_224',
    // Section 2.4 fields, so a freshly fetched model describes itself in the picker the
    // same way a converted one does instead of relying on the backend to infer them.
    displayName: 'MobileNetV2 (ImageNet)',
    description:
      'The bundled fallback: small, instant to load, and good enough for everyday items.',
    quantization: 'none',
    downloadBytes,
    version: '1.0.0',
    createdAt: new Date().toISOString(),
    baseModel: 'MobileNetV2',
    inputSize: 224,
    inputRange: [0, 1],
    outputActivation: 'logits',
    classOffset: 1,
    labelKind: 'imagenet',
    classes: null,
    classCount: 1001,
    notes:
      'Pretrained ImageNet-1k MobileNetV2 graph model (TF-Hub conversion). Load with ' +
      'tf.loadGraphModel, not loadLayersModel. Feed pixels scaled to [0,1] and nothing ' +
      'else: the graph itself contains hub_input/Mul (y=2.0) followed by hub_input/Sub ' +
      '(y=1.0), which rescales [0,1] to [-1,1] internally, so the Keras x/127.5-1 ' +
      'transform would double-apply. The output is 1001 raw logits with no softmax; ' +
      'index 0 is the TF-Slim synthetic background class, so slice it off (classOffset ' +
      '1) before softmax and the remaining 1000 indices line up with IMAGENET_CLASSES.',
  };
}

function buildSourceText(fetchedAt) {
  return [
    'EcoSort - pretrained fallback model provenance',
    '=============================================',
    '',
    `Upstream URL : ${BASE_URL}`,
    'Files        : model.json, group1-shard1of4 .. group1-shard4of4',
    `Total shards : ${KNOWN_TOTAL_SHARD_BYTES} bytes`,
    'Architecture : MobileNetV2, depth multiplier 1.0, 224x224 input, ImageNet-1k (1001 logits)',
    'Format       : TensorFlow.js graph model (converted TF-Hub SavedModel)',
    '',
    'Licence      : Apache License, Version 2.0.',
    '               The underlying TF-Hub module',
    '               https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification',
    '               is published by Google under the Apache License 2.0:',
    '               http://www.apache.org/licenses/LICENSE-2.0',
    '               The weights are redistributed here unmodified, for offline inference.',
    '',
    `Fetched at   : ${fetchedAt}`,
    'Fetched by   : scripts/fetch-mobilenet.mjs',
    '',
  ].join('\n');
}

await runFetcher({
  name: 'MobileNetV2',
  title: 'MobileNetV2 (1.0, 224)',
  scriptName: 'fetch-mobilenet.mjs',
  baseUrl: BASE_URL,
  dirName: 'mobilenet_v2',
  remoteFiles: [
    'model.json',
    'group1-shard1of4',
    'group1-shard2of4',
    'group1-shard3of4',
    'group1-shard4of4',
  ],
  knownTotalBytes: KNOWN_TOTAL_SHARD_BYTES,
  buildMetadata,
  buildSourceText,
  failureNote: 'The pretrained fallback model was NOT installed.',
});
