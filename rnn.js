
const fs = require('fs');
const path = require('path');

class RNN {
  constructor(options = {}) {
    this.vocabSize = options.vocabSize;
    this.hiddenSize = options.hiddenSize ?? 64;
    this.learningRate = options.learningRate ?? 0.01;
    this.clipValue = options.clipValue ?? 5;

    if (!Number.isInteger(this.vocabSize) || this.vocabSize < 2) {
      throw new Error('vocabSize must be an integer of at least 2.');
    }

    this.inputSize = this.vocabSize;
    this.outputSize = this.vocabSize;

    this.initWeights();

    this.chordVocabulary = options.chordVocabulary || [];

this.chordToId = new Map(
  this.chordVocabulary.map((chord, id) => [chord, id])
);
  }

  // ---------------------------------------------------------
  // Xavier uniform initialization
  // ---------------------------------------------------------

  initWeights() {
    this.Wxh = RNN.xavierUniformMatrix(
      this.hiddenSize,
      this.inputSize
    );

    this.Whh = RNN.xavierUniformMatrix(
      this.hiddenSize,
      this.hiddenSize
    );

    this.Why = RNN.xavierUniformMatrix(
      this.outputSize,
      this.hiddenSize
    );

    this.bh = RNN.zeros(this.hiddenSize);
    this.by = RNN.zeros(this.outputSize);
  }

  static xavierUniformMatrix(rows, cols) {
    const limit = Math.sqrt(6 / (rows + cols));

    return Array.from(
      { length: rows },
      () =>
        Array.from(
          { length: cols },
          () => (Math.random() * 2 - 1) * limit
        )
    );
  }

  static zeros(n) {
    return new Array(n).fill(0);
  }

  static zeroMatrix(rows, cols) {
    return Array.from(
      { length: rows },
      () => new Array(cols).fill(0)
    );
  }

  static matVecMul(matrix, vector) {
    const result = new Array(matrix.length).fill(0);

    for (let i = 0; i < matrix.length; i++) {
      let sum = 0;

      for (let j = 0; j < matrix[i].length; j++) {
        sum += matrix[i][j] * vector[j];
      }

      result[i] = sum;
    }

    return result;
  }

  static matTVecMul(matrix, vector) {
    const cols = matrix[0].length;
    const result = new Array(cols).fill(0);

    for (let i = 0; i < matrix.length; i++) {
      for (let j = 0; j < cols; j++) {
        result[j] += matrix[i][j] * vector[i];
      }
    }

    return result;
  }

  static addVec(a, b) {
    return a.map((value, i) => value + b[i]);
  }

  static tanhVec(vector) {
    return vector.map(value => Math.tanh(value));
  }

  static addOuter(matrix, a, b) {
    for (let i = 0; i < a.length; i++) {
      for (let j = 0; j < b.length; j++) {
        matrix[i][j] += a[i] * b[j];
      }
    }
  }

  static softmax(logits) {
    const maxLogit = Math.max(...logits);

    const exps = logits.map(
      value => Math.exp(value - maxLogit)
    );

    const sum = exps.reduce((a, b) => a + b, 0);

    return exps.map(value => value / sum);
  }

  // ---------------------------------------------------------
  // Chord encoding
  // ---------------------------------------------------------

  encodeChord(chordId) {
    const vector = RNN.zeros(this.vocabSize);
    vector[chordId] = 1;

    return vector;
  }

  // ---------------------------------------------------------
  // Forward propagation
  // ---------------------------------------------------------

  forward(chordIds) {
    const xs = [];
    const hs = [];
    const ps = [];

    const initialHidden = RNN.zeros(this.hiddenSize);

    for (let t = 0; t < chordIds.length; t++) {
      const x = this.encodeChord(chordIds[t]);

      const previousHidden =
        t === 0 ? initialHidden : hs[t - 1];

      const inputContribution = RNN.matVecMul(
        this.Wxh,
        x
      );

      const recurrentContribution = RNN.matVecMul(
        this.Whh,
        previousHidden
      );

      const raw = RNN.addVec(
        RNN.addVec(
          inputContribution,
          recurrentContribution
        ),
        this.bh
      );

      const h = RNN.tanhVec(raw);

      const logits = RNN.addVec(
        RNN.matVecMul(this.Why, h),
        this.by
      );

      const probabilities = RNN.softmax(logits);

      xs.push(x);
      hs.push(h);
      ps.push(probabilities);
    }

    return {
      xs,
      hs,
      ps,
      chordIds,
      length: chordIds.length
    };
  }

  // ---------------------------------------------------------
  // Cross-entropy loss
  // ---------------------------------------------------------

  crossEntropyLoss(probabilities, targets) {
    if (targets.length === 0) {
      return 0;
    }

    let loss = 0;

    for (let t = 0; t < targets.length; t++) {
      const probability = Math.max(
        probabilities[t][targets[t]],
        1e-12
      );

      loss -= Math.log(probability);
    }

    return loss / targets.length;
  }

  // ---------------------------------------------------------
  // Gradient initialization
  // ---------------------------------------------------------

  zeroGradients() {
    return {
      dWxh: RNN.zeroMatrix(
        this.hiddenSize,
        this.inputSize
      ),

      dWhh: RNN.zeroMatrix(
        this.hiddenSize,
        this.hiddenSize
      ),

      dWhy: RNN.zeroMatrix(
        this.outputSize,
        this.hiddenSize
      ),

      dbh: RNN.zeros(this.hiddenSize),
      dby: RNN.zeros(this.outputSize)
    };
  }

  // ---------------------------------------------------------
  // Backpropagation through time
  // ---------------------------------------------------------

  backPropagateThroughTime(cache, targets) {
    const grads = this.zeroGradients();

    let dhNext = RNN.zeros(this.hiddenSize);

    for (let t = targets.length - 1; t >= 0; t--) {
      const dLogits = cache.ps[t].slice();

      dLogits[targets[t]] -= 1;

      RNN.addOuter(
        grads.dWhy,
        dLogits,
        cache.hs[t]
      );

      for (let i = 0; i < this.outputSize; i++) {
        grads.dby[i] += dLogits[i];
      }

      const dh = RNN.addVec(
        RNN.matTVecMul(this.Why, dLogits),
        dhNext
      );

      const dRaw = cache.hs[t].map(
        (h, i) => (1 - h * h) * dh[i]
      );

      for (let i = 0; i < this.hiddenSize; i++) {
        grads.dbh[i] += dRaw[i];
      }

      RNN.addOuter(
        grads.dWxh,
        dRaw,
        cache.xs[t]
      );

      const previousHidden =
        t === 0
          ? RNN.zeros(this.hiddenSize)
          : cache.hs[t - 1];

      RNN.addOuter(
        grads.dWhh,
        dRaw,
        previousHidden
      );

      dhNext = RNN.matTVecMul(
        this.Whh,
        dRaw
      );
    }

    const divisor = Math.max(1, targets.length);

    for (const key of ['dWxh', 'dWhh', 'dWhy']) {
      for (let i = 0; i < grads[key].length; i++) {
        for (let j = 0; j < grads[key][i].length; j++) {
          grads[key][i][j] /= divisor;
        }
      }
    }

    for (const key of ['dbh', 'dby']) {
      for (let i = 0; i < grads[key].length; i++) {
        grads[key][i] /= divisor;
      }
    }

    return grads;
  }

  // ---------------------------------------------------------
  // Weight updates using gradient descent
  // ---------------------------------------------------------

  updateWeights(grads) {
    const clamp = value =>
      Math.max(
        -this.clipValue,
        Math.min(this.clipValue, value)
      );

    const matrices = [
      ['Wxh', 'dWxh'],
      ['Whh', 'dWhh'],
      ['Why', 'dWhy']
    ];

    for (const [weightName, gradientName] of matrices) {
      const weights = this[weightName];
      const gradient = grads[gradientName];

      for (let i = 0; i < weights.length; i++) {
        for (let j = 0; j < weights[i].length; j++) {
          weights[i][j] -=
            this.learningRate * clamp(gradient[i][j]);
        }
      }
    }

    const vectors = [
      ['bh', 'dbh'],
      ['by', 'dby']
    ];

    for (const [weightName, gradientName] of vectors) {
      for (let i = 0; i < this[weightName].length; i++) {
        this[weightName][i] -=
          this.learningRate * clamp(grads[gradientName][i]);
      }
    }
  }

  // ---------------------------------------------------------
  // Training
  // ---------------------------------------------------------

  train(sequences, options = {}) {
    const epochs = options.epochs ?? 100;
    const verbose = options.verbose !== false;

    const history = [];

    for (let epoch = 0; epoch < epochs; epoch++) {
      const shuffled = sequences.slice();

      for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));

        [shuffled[i], shuffled[j]] = [
          shuffled[j],
          shuffled[i]
        ];
      }

      let totalLoss = 0;
      let totalTargets = 0;

      for (const sequence of shuffled) {
        if (sequence.length < 2) {
          continue;
        }

        const inputs = sequence.slice(0, -1);
        const targets = sequence.slice(1);

        const cache = this.forward(inputs);

        const loss = this.crossEntropyLoss(
          cache.ps,
          targets
        );

        const grads = this.backPropagateThroughTime(
          cache,
          targets
        );

        this.updateWeights(grads);

        totalLoss += loss * targets.length;
        totalTargets += targets.length;
      }

      const averageLoss =
        totalTargets > 0
          ? totalLoss / totalTargets
          : 0;

      history.push(averageLoss);

      if (
        verbose &&
        (
          epoch === 0 ||
          (epoch + 1) % 10 === 0 ||
          epoch === epochs - 1
        )
      ) {
        console.log(
          `Epoch ${epoch + 1}/${epochs} | ` +
          `Loss: ${averageLoss.toFixed(4)}`
        );
      }
    }

    return history;
  }

  // ---------------------------------------------------------
  // Next-chord prediction
  // ---------------------------------------------------------

  predictNext(chordIds, options = {}) {
    if (!Array.isArray(chordIds) || chordIds.length === 0) {
      throw new Error(
        'Provide at least one chord ID to predict the next chord.'
      );
    }

    const cache = this.forward(chordIds);

    let probabilities =
      cache.ps[cache.ps.length - 1].slice();

    const temperature = options.temperature ?? 1;

    if (temperature <= 0) {
      throw new Error(
        'temperature must be greater than zero.'
      );
    }

    if (temperature !== 1) {
      const logits = probabilities.map(
        p => Math.log(Math.max(p, 1e-12)) / temperature
      );

      probabilities = RNN.softmax(logits);
    }

    let chordId;

    if (options.sample) {
      const r = Math.random();

      let cumulative = 0;

      chordId = probabilities.length - 1;

      for (let i = 0; i < probabilities.length; i++) {
        cumulative += probabilities[i];

        if (r <= cumulative) {
          chordId = i;
          break;
        }
      }
    } else {
      chordId = probabilities.indexOf(
        Math.max(...probabilities)
      );
    }

    return {
      chordId,
      probability: probabilities[chordId],
      probabilities
    };
  }

predictWithSequence(chords, options = {}) {
  if (!Array.isArray(chords) || chords.length === 0) {
    throw new Error("Provide at least one chord.");
  }

  if (!this.chordToId || this.chordToId.size === 0) {
    throw new Error("Chord vocabulary has not been initialized.");
  }

  const chordIds = chords.map(chord => {
    const chordId = this.chordToId.get(chord);

    if (chordId === undefined) {
      throw new Error(`Unknown chord: ${chord}`);
    }

    return chordId;
  });

  const prediction = this.predictNext(chordIds, options);

  return {
    chord: this.chordVocabulary[prediction.chordId],
    chordId: prediction.chordId,
    probability: prediction.probability,
    probabilities: prediction.probabilities.map(
      (probability, id) => ({
        chord: this.chordVocabulary[id],
        probability
      })
    )
  };
}
  // ---------------------------------------------------------
  // Save model
  // ---------------------------------------------------------

  save(filePath, chordVocabulary) {
    const payload = {
      config: {
        vocabSize: this.vocabSize,
        hiddenSize: this.hiddenSize,
        learningRate: this.learningRate,
        clipValue: this.clipValue
      },

      chordVocabulary,

      weights: {
        Wxh: this.Wxh,
        Whh: this.Whh,
        Why: this.Why,
        bh: this.bh,
        by: this.by
      }
    };

    fs.writeFileSync(
      filePath,
      JSON.stringify(payload, null, 2),
      'utf8'
    );
  }
}

// ---------------------------------------------------------
// Dataset loading
// ---------------------------------------------------------

function loadSongs(filePath) {
  const raw = JSON.parse(
    fs.readFileSync(filePath, 'utf8')
  );

  const songs = Array.isArray(raw) ? raw : [raw];

  const validSongs = songs.filter(
    song =>
      song &&
      Array.isArray(song.chords) &&
      song.chords.length >= 2
  );

  if (validSongs.length === 0) {
    throw new Error(
      'No valid songs found. songs.json must contain ' +
      'song objects with at least two chords.'
    );
  }

  for (const song of validSongs) {
    song.chords = song.chords
      .filter(
        chord =>
          typeof chord === 'string' &&
          chord.trim().length > 0
      )
      .map(chord => chord.trim());
  }

  return validSongs.filter(
    song => song.chords.length >= 2
  );
}

//main

function main() {
  const datasetPath = path.join(
    __dirname,
    'songs.json'
  );

  if (!fs.existsSync(datasetPath)) {
    throw new Error(
      `Could not find ${datasetPath}. Put songs.json beside this script.`
    );
  }

  const songs = loadSongs(datasetPath);

  const chordVocabulary = [
    ...new Set(
      songs.flatMap(song => song.chords)
    )
  ].sort();

  if (chordVocabulary.length < 2) {
    throw new Error(
      'The dataset must contain at least two distinct chords.'
    );
  }

  const chordToId = new Map(
    chordVocabulary.map(
      (chord, id) => [chord, id]
    )
  );

  const sequences = songs.map(
    song =>
      song.chords.map(
        chord => chordToId.get(chord)
      )
  );

  const totalTransitions = sequences.reduce(
    (sum, sequence) =>
      sum + Math.max(0, sequence.length - 1),
    0
  );

  console.log('======================================');
  console.log('     NEXT-CHORD PREDICTION RNN');
  console.log('======================================');

  console.log(`Songs loaded: ${songs.length}`);
  console.log(`Distinct chords: ${chordVocabulary.length}`);
  console.log(`Training transitions: ${totalTransitions}`);
  console.log('Weight initialization: Xavier uniform');
  console.log('');

  const model = new RNN({
    vocabSize: chordVocabulary.length,
    hiddenSize: Number(process.env.HIDDEN_SIZE || 64),
    learningRate: Number(process.env.LEARNING_RATE || 0.01),
    clipValue: 5
  });

  const epochs = Number(process.env.EPOCHS || 500);

  console.log('Starting training...\n');

  const history = model.train(
    sequences,
    {
      epochs,
      verbose: true
    }
  );

  const modelPath = path.join(
    __dirname,
    'rnn_chord_model.json'
  );

  model.save(modelPath, chordVocabulary);

  console.log('\n======================================');
  console.log('           TRAINING RESULTS');
  console.log('======================================');

  console.log(
    `Initial loss: ${history[0].toFixed(4)}`
  );

  console.log(
    `Final loss: ${history[history.length - 1].toFixed(4)}`
  );

  console.log(`Saved model: ${modelPath}`);

  const demoSong = songs[0];

  const promptChords = demoSong.chords.slice(
    0,
    Math.min(5, demoSong.chords.length)
  );

  const promptIds = promptChords.map(
    chord => chordToId.get(chord)
  );

  const prediction = model.predictNext(promptIds);

  const predictedChord =
    chordVocabulary[prediction.chordId];

  console.log('\n======================================');
  console.log('        NEXT-CHORD PREDICTION');
  console.log('======================================');

  console.log(
    `Example song: ${demoSong.songId ?? '(no songId)'}`
  );

  console.log(
    `Input chords: ${promptChords.join(' -> ')}`
  );

  console.log(
    `Predicted next chord: ${predictedChord}`
  );

  console.log(
    `Model probability: ${(prediction.probability * 100).toFixed(2)}%`
  );

  if (demoSong.chords.length > promptChords.length) {
    console.log(
      `Actual next chord: ${demoSong.chords[promptChords.length]}`
    );
  }

  console.log('\nTraining completed.');
}

if (require.main === module) {
  try {
    main();
  } catch (error) {
    console.error(`Error: ${error.message}`);
    process.exitCode = 1;
  }
}

module.exports = RNN;